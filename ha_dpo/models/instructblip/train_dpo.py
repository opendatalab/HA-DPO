import os
os.environ["WANDB_PROJECT"]="ha-dpo"

import yaml
import json
import copy
import torch
import random
import argparse
import numpy as np
from PIL import Image
from argparse import Namespace
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from torch.utils.data import ConcatDataset

from peft.peft_model import PeftModelForCausalLM
from transformers import TrainerCallback
from transformers import HfArgumentParser, TrainingArguments

import vigc.tasks as tasks
from vigc.common.config import Config

from ha_dpo.trainer.instructblip_dpo_trainer import InstructBLIPDPOTrainer
from ha_dpo.models.instructblip.dpo_dataset import PopeDataset, AugmentedCaptionDataset


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    
    # model config path
    cfg_path: str = field(metadata={"help": "path to configuration file."})
    
    # data parameters
    desc_train_data_path: Optional[str] = field(default=None, metadata={"help": "path of the description positive-negative data."})
    pope_train_data_path: Optional[str] = field(default=None, metadata={"help": "path of the pope-format positive-negative data."})
    vg_path: Optional[str] = field(default="", metadata={"help": "path of visual genome annotation file."})
    
    # hyper-parameters
    seed: Optional[int] = field(default=42, metadata={"help": "training and data seed."})
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    
    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "maximum value of gradient norm"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=True, metadata={"help": "whether to find unused parameters. set to False when `gradient_checkpointing` is False."}
    )
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[float] = field(default=-1, metadata={"help": "number of trained eppchs."})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=-1, metadata={"help": "the saving frequency"})
    evaluation_strategy: Optional[str] = field(default='no', metadata={"help": "the evaluation strategy"})
    eval_steps: Optional[float] = field(default=None, metadata={"help": "the evaluation frequency"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    run_name: Optional[str] = field(default="dpo_instructblip", metadata={"help": "name of the run"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    
    # lora parameters
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=2, metadata={"help": "the lora r parameter"})
    lora_target_modules: Optional[list[str]] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"], metadata={"help": "the lora modules"})
    freeze_llm_proj: Optional[bool] = field(default=True, metadata={"help": "whether to freeze llama_proj module"})
    
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


# callback used to save model
# !HACK: wandb upload failed!
class MyCallback(TrainerCallback):
    "A callback that prints a message at the end of training"
    def on_train_end(self, args, state, control, **kwargs):
        # save model
        if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
            print("Save model in the end of training")
            with open(os.path.join(args.output_dir, "training_args.yaml"), "w") as f:
                yaml.dump(args, f)
            # save lora weights
            if isinstance(kwargs['model'].llm_model, PeftModelForCausalLM):
                kwargs['model'].llm_model.save_pretrained(args.output_dir)
    
    
def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    cfg_dict = {'cfg_path': script_args.cfg_path, 'options': None}
    cfg = Config(Namespace(**cfg_dict))
    cfg.pretty_print()
    
    # set dpo model parameters
    cfg.config.model.lora_config.lora_r = script_args.lora_r
    cfg.config.model.lora_config.lora_alpha = script_args.lora_alpha
    cfg.config.model.lora_config.lora_dropout = script_args.lora_dropout
    cfg.config.model.lora_config.lora_target_modules = script_args.lora_target_modules
    cfg.config.model.freeze_llm_proj = script_args.freeze_llm_proj

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    tokenizer = model.llm_tokenizer
    
    # build reference model
    ref_cfg = copy.deepcopy(cfg)
    ref_cfg.config.model.lora_config.lora_r = 0
    ref_task = tasks.setup_task(ref_cfg)
    ref_model = task.build_model(ref_cfg)
    for n,p in ref_model.named_parameters():
        p.requires_grad = False
    
    if script_args.desc_train_data_path is not None:
        desc_train_dataset = AugmentedCaptionDataset(
            data_path = script_args.desc_train_data_path,
            vg_path = script_args.vg_path,
            cfg = cfg.config,
            seed = script_args.seed,
        )
    if script_args.pope_train_data_path is not None:
        pope_train_dataset = PopeDataset(
            data_path = script_args.pope_train_data_path,
            vg_path = script_args.vg_path,
            cfg = cfg.config,
        )
    if script_args.pope_train_data_path and script_args.desc_train_data_path:
        train_dataset = ConcatDataset([desc_train_dataset]+[pope_train_dataset]*2)  # keep data ratio as pope:desc=2:1
    elif script_args.pope_train_data_path:
        train_dataset = pope_train_dataset
    elif script_args.desc_train_dataset:
        train_dataset = desc_train_dataset
    
    # if not use gradient_checkpointing, do not set ddp_find_unused_parameters
    if not script_args.gradient_checkpointing:
        script_args.ddp_find_unused_parameters = False
    
    # initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        ddp_find_unused_parameters=script_args.ddp_find_unused_parameters,
        learning_rate=script_args.learning_rate,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        max_grad_norm=script_args.max_grad_norm,
        seed=script_args.seed,
    )
    
    # initialize the DPO trainer
    dpo_trainer = InstructBLIPDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )
    
    # model save callback
    dpo_trainer.add_callback(MyCallback())
    
    dpo_trainer.train()
    
    # save script args
    with open(os.path.join(training_args.output_dir, "script_args.yaml"), "w") as f:
        yaml.dump(script_args, f)
    
if __name__ == "__main__":
    main()