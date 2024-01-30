import os
import tqdm
import json
import argparse
from PIL import Image
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.conversation.conversation import Chat, CONV_VISION_LLama2

import logging
logging.basicConfig(level = logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="POPE Evaluation")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--coco-path", required=True, help="path to COCO2014 images.")
    parser.add_argument("--pope-path", required=True, help="path to POPE annotation file.")
    parser.add_argument("--set", required=True, help="which set of POPE, choose between random/popular/adv.")
    parser.add_argument("--llama-model", default=None, help="path to language model file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def ddp_model(model, cfg):
    """
        A property to get the DDP-wrapped model on the device.
    """
   # distributed training wrapper
    if cfg.run_cfg.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        _wrapped_model = DDP(
            model, device_ids=[local_rank], output_device=local_rank
        )
    else:
        _wrapped_model = model

    return _wrapped_model

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
    
    
def convert_dict_to_tensor(results, device):
    part_tensor = json.dumps(results)
    part_tensor = torch.Tensor([ord(part_tensor[i]) for i in range(len(part_tensor))]).long().to(device)
    return part_tensor
    
# ========================================
#             Model Initialization
# ========================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info('Initializing Chat')
    args = parse_args()
    cfg = Config(args)
    if args.llama_model is not None:
        cfg.model_cfg.llama_model = args.llama_model

    init_distributed_mode(cfg.run_cfg)

    model_config = cfg.model_cfg
    if args.llama_model is not None:
        model_config.llama_model = args.llama_model

    print(f"llama model: {cfg.model_cfg.llama_model}")

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    # move to device
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    model = model.to(device)
    model = ddp_model(model, cfg)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    if isinstance(model, DDP):
        chat = Chat(model.module, vis_processor, device=device)
    else:
        chat = Chat(model, vis_processor, device=device)

    CONV = CONV_VISION_LLama2

    logging.info('Initialization Finished')

    outputs_all = []
    if args.set == "random":
        questions_file = open(os.path.join(args.pope_path, "output/coco/coco_pope_random.json"), "r")
    if args.set == "popular":
        questions_file = open(os.path.join(args.pope_path, "output/coco/coco_pope_popular.json"), "r")
    if args.set == "adv":
        questions_file = open(os.path.join(args.pope_path, "output/coco/coco_pope_adversarial.json"), "r")
    lines = list(questions_file.readlines())

    rank, word_size = get_rank(), get_world_size()
    step = len(lines) // word_size + 1
    start, end = rank * step, (rank + 1) * step

    results = []
    if get_rank() == 0:
        logging.info("generating answers...")

    for line in tqdm.tqdm(lines[start:end]):
        data = json.loads(line)
        message_input = data["text"]
        image = data["image"]
        question_id = data["question_id"]

        image = os.path.join(args.coco_path, "val2014", image)
        # model inference
        # reset
        img_list = []
        if image is not None:
            if isinstance(image, str):  # is a image path
                raw_image = Image.open(image).convert('RGB')
                image = chat.vis_processor(raw_image).unsqueeze(0).to(device)
            elif isinstance(image, Image.Image):
                raw_image = image
                image = chat.vis_processor(raw_image).unsqueeze(0).to(device)
            elif isinstance(image, torch.Tensor):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                image = image.to(device)

            image_emb, _ = chat.model.encode_img(image)
            img_list.append(image_emb)
            conv = CONV_VISION_LLama2.copy()
            conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        else:
            conv = CONV_VISION_LLama2.copy()

        # ask model
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
            and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], message_input])
        else:
            conv.append_message(conv.roles[0], message_input)

        output_text = chat.answer(conv=conv, img_list=img_list, temperature=1.0, length_penalty=1.0, do_sample=False)[0]

        results.append({
            "question": message_input,
            "answer": output_text,
            "question_id": question_id,
        })

    # convert dictionary -> tensor for gather all results in all ranks
    part_tensor = convert_dict_to_tensor(results, device)
    shape_tensor = torch.tensor(part_tensor.shape, device=device)
    shape_list = [shape_tensor.clone() for _ in range(get_world_size())]
    dist.all_gather(shape_list, shape_tensor)

    # gather tensor
    max_shape = max(shape_list)
    part_tensor_pad = torch.zeros(max_shape).to(device)
    part_tensor_pad[:part_tensor.shape[0]] = part_tensor
    tensor_list = [part_tensor_pad.clone() for _ in range(get_world_size())]
    dist.all_gather(tensor_list, part_tensor_pad)

    if dist.get_rank() == 0:
        results_all_rank = []
        for tensor, shape in zip(tensor_list, shape_list):
            t = tensor.long()[:shape]
            _data = "".join([chr(t[i].item()) for i in range(t.shape[0])])
            _data = json.loads(_data)
            results_all_rank.extend(_data)
        # sort according to question_id
        results_all_rank = sorted(results_all_rank, key=lambda x:x["question_id"])
        
        with open(f"ha_dpo/models/minigpt4/pope_{args.set}.jsonl", "w") as f:
            for res in results_all_rank:
                f.write(json.dumps(res)+'\n')