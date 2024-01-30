import os
import json
import tqdm
import torch
import random
import argparse
import datetime
import numpy as np
from PIL import Image

import vigc.tasks as tasks
from vigc.common.config import Config
from vigc.common.dist_utils import get_rank, init_distributed_mode
from vigc.common.logger import setup_logger
from vigc.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)

from vigc.tasks import *
from vigc.models import *
from vigc.runners import *
from vigc.processors import *
from vigc.datasets.builders import *
from vigc.common.utils import now
from vigc.common.registry import registry

def parse_args():
    parser = argparse.ArgumentParser(description="POPE evaluation")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--pope-path", required=True, help="path to POPE.")
    parser.add_argument("--coco-path", required=True, help="path to coco2014.")
    parser.add_argument("--set", required=True, help="POPE set (random, popular, adversarial)")
    parser.add_argument("--llm-model", type=str, default=None, required=False, help="path to language model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args

def convert_dict_to_tensor(results, device):
    part_tensor = json.dumps(results)
    part_tensor = torch.Tensor([ord(part_tensor[i]) for i in range(len(part_tensor))]).long().to(device)
    return part_tensor


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    cfg = Config(args)
    if args.llm_model is not None:
        cfg.config.model.llm_model = args.llm_model
    init_distributed_mode(cfg.run_cfg)

    cfg.pretty_print()
    
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    vis_processor_cfg = cfg.datasets_cfg.instruct_blip_given_q_coco2017_vig_test.vis_processor.eval
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    outputs_all = []
    if args.set == "random":
        questions_file = open(os.path.join(args.pope_path, "output/coco/coco_pope_random.json"), "r")
    if args.set == "popular":
        questions_file = open(os.path.join(args.pope_path, "output/coco/coco_pope_popular.json"), "r")
    if args.set == "adv":
        questions_file = open(os.path.join(args.pope_path, "output/coco/coco_pope_adversarial.json"), "r")
    lines = list(questions_file.readlines())

    rank, word_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    step = len(lines) // word_size + 1
    start, end = rank * step, min((rank + 1) * step, len(lines))

    results = []
    if get_rank() == 0:
        print("generating answers...")
        
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    model = model.to(device)
        
    for idx in tqdm.tqdm(range(start,end)):
        line = lines[idx]
        data = json.loads(line)
        question_id = data["question_id"]
        message_input = data["text"]
        image = data["image"]
        image = os.path.join(args.coco_path, "val2014", image)
        image = Image.open(image).convert("RGB")
    
        # prepare the image
        image = vis_processor(image).unsqueeze(0).to(device)

        samples = {"prompt": message_input,"image": image}
        answer = model.generate(samples, num_beams=5)[0]
                        
        results.append({
            "question_id": question_id,
            "question": message_input,
            "answer": answer,
        })
        
    # convert dictionary -> tensor for gather all results in all ranks
    part_tensor = convert_dict_to_tensor(results, device)
    shape_tensor = torch.tensor(part_tensor.shape, device=device)
    shape_list = [shape_tensor.clone() for _ in range(int(os.environ["WORLD_SIZE"]))]
    torch.distributed.all_gather(shape_list, shape_tensor)

    # gather tensor
    max_shape = max(shape_list)
    part_tensor_pad = torch.zeros(max_shape).to(device)
    part_tensor_pad[:part_tensor.shape[0]] = part_tensor
    tensor_list = [part_tensor_pad.clone() for _ in range(int(os.environ["WORLD_SIZE"]))]
    torch.distributed.all_gather(tensor_list, part_tensor_pad)

    if int(os.environ["LOCAL_RANK"]) == 0:
        results_all_rank = []
        for tensor, shape in zip(tensor_list, shape_list):
            t = tensor.long()[:shape]
            _data = "".join([chr(t[i].item()) for i in range(t.shape[0])])
            _data = json.loads(_data)
            results_all_rank.extend(_data)
        # sort according to question_id
        res_file = f"./ha_dpo/models/instructblip/pope_{args.set}.jsonl"
        with open(res_file, "w") as f:
            for res in results_all_rank:
                f.write(json.dumps(res))
                f.write("\n")
    
if __name__ == "__main__":
    main()
