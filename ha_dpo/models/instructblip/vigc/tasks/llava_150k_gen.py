from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import torch
import logging
from vigc.models.blip2_models.blip2 import disabled_train
from vigc.models import load_model_and_preprocess
from peft import LoraConfig, get_peft_model

VIG_INSTRUCTIONS = {
    "comp":
        "Based on the given image, generate an in-depth reasoning question and then answer it.",
    "conv":
        "Generate a question based on the content of the given image and then answer it.",
    "desc":
        "Generate a question to describe the image content in detail and then answer it."
}


@registry.register_task("instruct_blip_llava_vig")
class InstructBlipLLavaVIGTask(BaseTask):

    def __init__(self, num_beams, max_len, min_len, use_nucleus_sampling, evaluate, task, report_metric=False,
                 answer_length=1, gen_style="vig", last_infer_all=False, in_section=False):
        super(InstructBlipLLavaVIGTask, self).__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.use_nucleus_sampling = use_nucleus_sampling
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.answer_length = answer_length
        task = task.lower()
        assert task in VIG_INSTRUCTIONS
        self.prompt = VIG_INSTRUCTIONS[task]
        gen_style = gen_style.lower()
        assert gen_style in ("vig", "vic")
        self.gen_style = gen_style
        self.last_infer_all = last_infer_all
        self.in_section = in_section

    def build_model_(self, cfg):
        model_config = cfg.model_cfg
        model, vis_processors, _ = load_model_and_preprocess(
            name=model_config.arch,
            model_type=model_config.model_type,
            is_eval=True,
        )

        model.load_checkpoint(model_config.pretrained)
        return model

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config)
        freeze_q_former = model_config.get("freeze_q_former", False)

        if freeze_q_former:
            for name, param in model.Qformer.named_parameters():
                param.requires_grad = False
            model.Qformer = model.Qformer.eval()
            model.Qformer.train = disabled_train
            logging.info("freeze Qformer")

            model.query_tokens.requires_grad = False
            logging.info("freeze Qformer query")
            print('Loading Q-Former Done')

        lora_config = model_config.get("lora_config")
        if lora_config is not None:
            lora_config = LoraConfig(
                r=lora_config.lora_r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias="none",  # won't use bias currently
                modules_to_save=[],  # TODO: might be helpful if save partial model
                task_type="CAUSAL_LM",
            )
            llm_model_lora = get_peft_model(model.llm_model, peft_config=lora_config)
            llm_model_lora.print_trainable_parameters()
            model.llm = llm_model_lora

            tuned_model_path = model_config.get("tuned_model")
            checkpoint = torch.load(tuned_model_path, map_location="cpu")["model"]

            model.load_state_dict(checkpoint, strict=False)
        return model

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        generate_cfg = run_cfg.generate_cfg

        num_beams = generate_cfg.num_beams
        max_len = generate_cfg.max_len
        min_len = generate_cfg.min_len
        use_nucleus_sampling = generate_cfg.get("use_nucleus_sampling", False)
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", False)
        answer_len = run_cfg.get("answer_length", 1)
        gen_style = run_cfg.get("gen_style", "vig")
        last_infer_all = run_cfg.get("last_infer_all", False)
        in_section = run_cfg.get("in_section", False)
        task = run_cfg.llava_task

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            use_nucleus_sampling=use_nucleus_sampling,
            evaluate=evaluate,
            report_metric=report_metric,
            answer_length=answer_len,
            task=task,
            gen_style=gen_style,
            last_infer_all=last_infer_all,
            in_section=in_section
        )

    def _update(self, conversation, text, step):
        last_flag = step == self.answer_length
        if conversation["question"] is None:  # update question and current text
            questions = []
            ori_answers = []
            for i, QA in enumerate(text):
                Q = None
                A = None
                if "Question:" in QA and "Answer:" in QA:
                    QA = QA.split("Question:")[-1].split("Answer:")
                    if len(QA) == 2:
                        Q = QA[0].strip()
                        A = QA[1].strip()
                questions.append(Q)
                ori_answers.append(A)
                if Q is None:
                    conversation["valid"][i] = False
            conversation["question"] = questions
            conversation["original_answers"] = ori_answers

            current_texts = []

            for i, (c, q) in enumerate(zip(conversation["instruction"], conversation["question"])):
                current_text = c
                if q:
                    current_text = f"{current_text} Question: {q} Answer:" if self.gen_style == "vig" else q
                current_texts.append(current_text)
            conversation["current_text"] = current_texts
        elif not self.in_section:
            current_answers = []
            if conversation["answers_given_question"] is None:
                conversation["answers_given_question"] = text
            for answer in text:
                A = ""
                if "." in answer:
                    A = answer.split(".")[0].strip() + "."
                if last_flag and self.last_infer_all:
                    A = answer
                current_answers.append(A)
            current_texts = []
            answers = []
            for i, (c, old_a, a) in enumerate(
                    zip(conversation["current_text"], conversation["answer"], current_answers)):
                current_text = f"{c} {a}".strip()
                current_texts.append(current_text)
                answer = f"{old_a} {a}".strip()
                answers.append(answer)
            conversation["current_text"] = current_texts
            conversation["answer"] = answers
        else:  # in_section
            current_answers = []
            first_flag = True
            if conversation["answers_given_question"] is None:
                conversation["answers_given_question"] = text
            else:
                first_flag = False
            for i, answer in enumerate(text):
                A = answer.split("\n\n")[0].strip()
                if last_flag and self.last_infer_all:
                    A = answer.strip()
                current_answers.append(A)
            current_texts = []
            answers = []
            for i, (c, old_a, a) in enumerate(
                    zip(conversation["current_text"], conversation["answer"], current_answers)):
                current_text = f"{c} {a}".strip() if first_flag else f"{c} \n\n{a}".strip()
                current_texts.append(current_text)
                conversation["answer_lst"][i].append(a)
                answer = f"{old_a} \n\n{a}".strip()
                answers.append(answer)
            conversation["current_text"] = current_texts
            conversation["answer"] = answers

        return conversation

    def valid_step(self, model, samples):
        results = []
        raw_samples = samples["raw_samples"]

        instructions = [self.prompt] * len(raw_samples)
        all_res = {
            "instruction": instructions,
            "current_text": instructions,
            "answer": [""] * len(instructions),
            "answer_lst": [list() for _ in instructions],
            "question": None,
            "original_answers": None,
            "answers_given_question": None,
            "valid": [True] * len(instructions)
        }
        images = samples["image"]

        for i in range(self.answer_length + 1):
            this_sample = {"prompt": all_res["current_text"], "image": images}
            answers = model.generate(
                this_sample,
                use_nucleus_sampling=False,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
            self._update(all_res, answers, step=i)

        for raw_samples, instruction, current_text, answer, question, valid, original_answer, corrected_answer, answer_lst in zip(
                raw_samples,
                all_res["instruction"],
                all_res["current_text"],
                all_res["answer"],
                all_res["question"],
                all_res["valid"],
                all_res["original_answers"],
                all_res["answers_given_question"],
                all_res["answer_lst"]):
            raw_samples = raw_samples.copy()

            raw_samples.update({
                "instruction": instruction,
                "whole_text": current_text,
                "answer": answer,
                "original_answer": original_answer,
                "answer_given_question": corrected_answer,
                "question": question,
                "answer_lst": answer_lst
            })
            if valid:
                results.append(raw_samples)

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        metrics = {"agg_metrics": 0.0}

        return metrics
