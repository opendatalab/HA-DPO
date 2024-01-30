import os
import json
import yaml
import tqdm
import torch
import argparse
from PIL import Image

from ha_dpo.shr_eval.shr_utils import *
from ha_dpo.shr_eval.gpt_utils import *

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_LLama2

def build_model(args, device):
    class model_args(argparse.Namespace):
        def __init__(self, cfg_path):
            self.cfg_path = cfg_path
            self.options = None
            
    model_args = model_args(cfg_path=args.cfg_path)
    cfg = Config(model_args)
    model_config = cfg.model_cfg
    if args.llama_model is not None:
        model_config.llama_model = args.llama_model    
    print(f"llama_model: {model_config.llama_model}")
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    model = model.to(device)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device=device)
    return chat

def dpo_model_inference(chat, image, prompt):
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

    message_input = prompt

    # ask model
    if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
        and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
        conv.messages[-1][1] = ' '.join([conv.messages[-1][1], message_input])
    else:
        conv.append_message(conv.roles[0], message_input)

    outputs = chat.answer(
        conv=conv, 
        img_list=img_list, 
        temperature=1.0, 
        num_beams=5,
        do_sample=False,
        length_penalty=1.0,
    )
    
    return outputs[0]

def parse_args():
    parser = argparse.ArgumentParser(description="SHR Evaluation")
    parser.add_argument("--api-key", type=str, required=True, help="key to the OPENAI API.")
    parser.add_argument("--cfg-path", type=str, required=True, help="path to configuration file.")
    parser.add_argument("--vg-path", type=str, required=True, help="path to vg file.")
    parser.add_argument("--shr-path", type=str, required=True, help="path to SHR annotation file.")
    parser.add_argument("--llama-model", type=str, default=None, help="path to configuration file.")
    parser.add_argument("--no-gpt-judge", default=False, action='store_true', help="whether not to do GPT evaluation. If True, only evaluate ngram repetition.")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    
    # setup openai
    setup_openai(args.api_key)
    
    config = yaml.safe_load(open(args.cfg_path))
    if args.llama_model is not None:
        config["model"]["llama_model"] = args.llama_model
    print(config)
    
    # build model
    device = f"cuda:{torch.cuda.current_device()}"
    chat = build_model(args, device)
    
    # visual genome annotations
    val_images = json.load(open(os.path.join(args.shr_path, "val_images_final.json")))
    vg_image_data = json.load(open(os.path.join(args.vg_path, "image_data.json")))
    id2path = {
        _data["image_id"]:os.path.join(args.vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
        for _data in vg_image_data
    }
    id2img = {_data["image_id"]:_data for _data in vg_image_data}
    region = json.load(open(os.path.join(args.vg_path, "region_descriptions.json")))
    id2reg = {r["regions"][0]["image_id"]:r for r in region}
    
    # factual information
    factual_inf = {}
    factual_part1 = os.path.join(args.shr_path, "shr_factual_part1.jsonl")
    factual_part2 = os.path.join(args.shr_path, "shr_factual_part2.jsonl")
    for line in open(factual_part1).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals
    for line in open(factual_part2).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals
    
    # whether to do 3 repeated evaluations or do only one evaluation
    judgement = {}
    run_all = ['run1']
    for run in run_all:
        judgement[run] = {}
    
    _gram1, _gram2, _gram3, _gram4 = 0, 0, 0, 0
    for _data in tqdm.tqdm(val_images):
        image_id = _data["image_id"]
        # ask model to describe the image
        prompt = "Describe this image in detail."
        image_path = id2path[int(image_id)]
        image = Image.open(image_path).convert("RGB")
        model_response = dpo_model_inference(chat, image, prompt)
        
        # get GPT judgement
        description = get_desc(id2img, id2reg, int(image_id))
        model_cap_sep, is_repeated = get_model_cap(model_response)
        # calculate repetition
        gram1 = cal_repetition(model_response,1)
        gram2 = cal_repetition(model_response,2)
        gram3 = cal_repetition(model_response,3)
        gram4 = cal_repetition(model_response,4)
        _gram1 += gram1
        _gram2 += gram2
        _gram3 += gram3
        _gram4 += gram4
            
        # skip gpt judgement 
        if args.no_gpt_judge:
            continue
            
        # GPT judgement
        factual_text = ""
        if str(image_id) in factual_inf:
            for text in factual_inf[str(image_id)]:
                factual_text += text
                factual_text += "\n"
        judge_prompt = GPT_JUDGE_PROMPT.format(description, factual_text, model_cap_sep)
        for run in run_all:
            while True:
                judge = get_gpt_response(prompt=judge_prompt)
                if "Judgement" in judge:
                    break
            # post-process
            final_judge = post_process_no_revise(judge, model_response)
            judgement[run][image_id] = {
                "raw_judgement": judge,
                "model_response": model_response,
                "judgement": final_judge,
            }
        
    whole_sample_cnt = len(val_images)
    if args.no_gpt_judge:
        print(f"gram-1 repetition: {round(_gram1/whole_sample_cnt, 3)}")
        print(f"gram-2 repetition: {round(_gram2/whole_sample_cnt, 3)}")
        print(f"gram-3 repetition: {round(_gram3/whole_sample_cnt, 3)}")
        print(f"gram-4 repetition: {round(_gram4/whole_sample_cnt, 3)}")
    else:
        base_eval_path = "./ha_dpo/models/minigpt4/shr_eval_results"
        if not os.path.exists(base_eval_path):
            os.mkdir(base_eval_path)
        localtime = time.asctime( time.localtime(time.time()) ).replace(' ', '_')
        # dump config file
        eval_path = os.path.join(base_eval_path, localtime)
        os.mkdir(eval_path)
        with open(os.path.join(base_eval_path, localtime, 'config.yaml'), "w") as f:
            yaml.dump(config, f)
        # save metrics
        metrics = {}
        for run in run_all:
            metrics[run] = {}
            get_metric(judgement[run], metrics[run])
        # repetition
        metrics['gram-1-repetition'] = round(_gram1/whole_sample_cnt, 3)
        metrics['gram-2-repetition'] = round(_gram2/whole_sample_cnt, 3)
        metrics['gram-3-repetition'] = round(_gram3/whole_sample_cnt, 3)
        metrics['gram-4-repetition'] = round(_gram4/whole_sample_cnt, 3)
        # halucination ratio
        metrics["mean_hal_ratio"] = round(
            sum(metrics[run]["hal_sents_ratio"] for run in run_all)/len(run_all), 3
        )
        # dump judgement file
        with open(os.path.join(base_eval_path, localtime, 'judgement.json'), "w") as f:
            json.dump(judgement, f)
        # dump metric file
        with open(os.path.join(base_eval_path, localtime, 'metrics.json'), "w") as f:
            json.dump(metrics, f)