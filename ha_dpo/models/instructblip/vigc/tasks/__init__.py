

from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
from vigc.tasks.caption_train_eval import InstructBlipCaptionTask
from vigc.tasks.llava_150k_gen import InstructBlipLLavaVIGTask
from vigc.tasks.vqa_train_eval import InstructBlipVQATask
from vigc.tasks.vqg_test import InstructBlipVQGTask
from vigc.tasks.dpo import DPOTask

def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "InstructBlipCaptionTask",
    "InstructBlipLLavaVIGTask",
    "InstructBlipVQATask",
    "InstructBlipVQGTask",
    "DPOTask",
]
