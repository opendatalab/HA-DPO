from vigc.common.dist_utils import main_process
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
from vigc.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
import json
import torch
import random
import os
import torch.distributed as dist
import logging
from vigc.models.blip2_models.blip2 import disabled_train
from peft import LoraConfig, get_peft_model

SAMPLE_PREFIX = {
    "okvqa": [('what', 5919),
              ('how', 580),
              ('is', 397),
              ('which', 335),
              ('where', 315),
              ('why', 246),
              ('who', 236),
              ('the', 132),
              ('when', 95),
              ('name', 86),
              ('in', 82),
              ('are', 79),
              ('can', 62),
              ('this', 59),
              ("what's", 55),
              ('if', 28),
              ('does', 28),
              ('would', 24),
              ('a', 20),
              ('these', 17),
              ('do', 15)],
    "a_okvqa": [('what', 10943),
                ('why', 1342),
                ('how', 971),
                ('which', 757),
                ('the', 731),
                ('where', 692),
                ('who', 428),
                ('in', 220),
                ('this', 149),
                ("what's", 83),
                ('if', 76),
                ('when', 63),
                ('these', 53),
                ('from', 43),
                ('for', 42),
                ('at', 33),
                ('to', 30),
                ('a', 20),
                ('on', 19),
                ('based', 18),
                ('during', 18)]
}


@registry.register_task("instruct_blip_vqg")
class InstructBlipVQGTask(BaseTask):

    def __init__(self, num_beams, max_len, min_len, use_nucleus_sampling, evaluate, report_metric=False,
                 generate_task="gq", weighted_sample=False, data_type=None):
        super(InstructBlipVQGTask, self).__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.use_nucleus_sampling = use_nucleus_sampling
        self.evaluate = evaluate

        self.report_metric = report_metric
        assert generate_task in ("gq", "ga")
        self.generate_task = generate_task

        if data_type:
            assert data_type in SAMPLE_PREFIX
            assert generate_task == "gq"  # 只有生成问题时才能用
        self.data_type = data_type
        self.weighted_sample = weighted_sample

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
        generate_task = run_cfg.get("generate_task", "gq")  # [gq or ga]

        report_metric = run_cfg.get("report_metric", False)
        data_type = run_cfg.get("data_type", None)
        weighted_sample = run_cfg.get("weighted_sample", False)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            use_nucleus_sampling=use_nucleus_sampling,
            evaluate=evaluate,
            report_metric=report_metric,
            generate_task=generate_task,
            data_type=data_type,
            weighted_sample=weighted_sample,
        )

    def gq_valid_step(self, model, samples):
        results = []
        raw_samples = samples["raw_samples"]
        prefix = []
        if self.data_type is not None:
            all_prefix = SAMPLE_PREFIX[self.data_type]
            if self.weighted_sample:
                prefix = random.choices(all_prefix, weights=[_[1] for _ in all_prefix], k=len(raw_samples))
            else:
                prefix = random.choices(all_prefix, k=len(raw_samples))
            prefix = [f" Question: {_[0]}" for _ in prefix]
            prompts = samples["prompt"]
            prompts = [f"{instruction}{question_prefix}" for instruction, question_prefix in zip(prompts, prefix)]
            samples["prompt"] = prompts
        answers = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len
        )
        updated_answers = []
        if self.data_type is not None:
            for pre, ans in zip(prefix, answers):
                update_answer = f"{pre} {ans}".strip()
                updated_answers.append(update_answer)
        if updated_answers:
            answers = updated_answers

        for raw_samples, answer in zip(raw_samples, answers):
            raw_samples = raw_samples.copy()
            raw_samples["question_answer"] = answer
            results.append(raw_samples)

        return results

    def ga_valid_step(self, model, samples):
        results = []
        raw_samples = samples["raw_samples"]

        answers, scores = model.generate_multi(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len
        )

        for raw_sample, answer, score in zip(raw_samples, answers, scores):
            raw_sample = raw_sample.copy()
            raw_sample["question_answer"] = answer
            raw_sample["weight"] = score.tolist()
            results.append(raw_sample)

        return results

    def valid_step(self, model, samples):
        if self.generate_task == "gq":
            return self.gq_valid_step(model, samples)
        else:
            return self.ga_valid_step(model, samples)

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file, eval_result = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        # eval_result = json.load(open(eval_result_file))

        if is_main_process():
            if self.generate_task == "gq":
                qa_result_file = os.path.join(registry.get_path("result_dir"), "question.json")
                result = []

                for d in eval_result:
                    image = d["image"]
                    QA = d["question_answer"]
                    if "Question:" in QA and "Answer:" in QA:
                        QA = QA.split("Question:")[-1].split("Answer:")
                        if len(QA) == 2:
                            Q, A = QA[0].strip(), QA[1].strip()
                            result.append({"image": image, "question": Q, "ori_answer": A})

                with open(qa_result_file, 'w') as f:
                    json.dump(result, f)
            else:
                qa_result_file = os.path.join(registry.get_path("result_dir"), "question_answer.json")
                result = []

                for d in eval_result:
                    ori_answers = d.pop("question_answer")
                    all_answers = []
                    for answer in ori_answers:
                        if answer.lower().startswith("answer"):
                            answer = answer.replace("answer:", "").replace("Answer:", "")
                            answer = answer.replace("answer :", "").replace("Answer :", "")
                            answer = answer.strip()
                        all_answers.append(answer)
                    this_res = d.copy()
                    this_res["answer"] = all_answers
                    result.append(this_res)

                with open(qa_result_file, 'w') as f:
                    json.dump(result, f)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        pass

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file, result
