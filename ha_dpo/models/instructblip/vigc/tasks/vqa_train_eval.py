from vigc.common.dist_utils import main_process
from mmpretrain.evaluation import VQAAcc
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
from peft import LoraConfig, get_peft_model
from vigc.models.blip2_models.blip2 import disabled_train
import os
import json
import logging


@registry.register_task("instruct_blip_vqa")
class InstructBlipVQATask(BaseTask):

    def __init__(self, num_beams, max_len, min_len, use_nucleus_sampling, evaluate, report_metric=True,
                 predict_by_rank=False):
        super(InstructBlipVQATask, self).__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.use_nucleus_sampling = use_nucleus_sampling
        self.evaluate = evaluate
        self.predict_by_rank = predict_by_rank

        self.report_metric = report_metric

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config)
        freeze_q_former = model_config.get("freeze_q_former", False)
        freeze_llm_proj = model_config.get("freeze_llm_proj", False)
        lora_config = model_config.get("lora_config", None)
        if lora_config is None:
            assert not freeze_q_former, "When lora is not used, you mustn't freeze Q-former"

        if freeze_q_former:
            for name, param in model.Qformer.named_parameters():
                param.requires_grad = False
            model.Qformer = model.Qformer.eval()
            model.Qformer.train = disabled_train
            logging.info("freeze Qformer")

            model.query_tokens.requires_grad = False
            logging.info("freeze Qformer query")
            print('Freeze Q-Former Done')

        if freeze_llm_proj:
            for name, param in model.llm_proj.named_parameters():
                param.requires_grad = False
            model.llm_proj = model.llm_proj.eval()
            model.llm_proj.train = disabled_train
            logging.info("freeze llm_proj")
            print("Freeze llm_proj Done")

        if lora_config is not None:
            print("Loading Lora ...")
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

        report_metric = run_cfg.get("report_metric", True)
        predict_by_rank = run_cfg.get("predict_by_rank", False)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            use_nucleus_sampling=use_nucleus_sampling,
            evaluate=evaluate,
            report_metric=report_metric,
            predict_by_rank=predict_by_rank
        )

    def valid_step(self, model, samples):
        results = []
        raw_samples = samples["raw_samples"]
        if self.predict_by_rank:
            answers = model.predict_by_rank(samples)
            for raw_sample in raw_samples:
                raw_sample["gt_answer"] = raw_sample.pop("gt_answer_rank", "")
                raw_sample["gt_answer_weight"] = raw_sample.pop("gt_answer_weight_rank", [])
        else:
            answers = model.generate(
                samples,
                use_nucleus_sampling=self.use_nucleus_sampling,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
        for raw_samples, answer in zip(raw_samples, answers):
            raw_samples = raw_samples.copy()
            answer = answer.strip()
            if answer.lower().startswith("answer"):
                answer = answer.replace("answer:", "").replace("Answer:", "")
                answer = answer.replace("answer :", "").replace("Answer :", "")
                answer = answer.strip()
            raw_samples["pred_answer"] = answer
            results.append(raw_samples)

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="question_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        vqaeval = VQAAcc()
        with open(eval_result_file) as f:
            results = json.load(f)
        acc = vqaeval.compute_metrics(results)["acc"]
        log_stats = {split_name: {"acc": acc}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {"agg_metrics": acc}
        return res
