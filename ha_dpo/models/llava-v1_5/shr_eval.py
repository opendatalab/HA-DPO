import os
import json
import tqdm
import torch
import argparse
from PIL import Image

from ha_dpo.shr_eval.shr_utils import *
from ha_dpo.shr_eval.gpt_utils import *

from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def main(args):
    # Model
    disable_torch_init()

    # setup openai
    setup_openai(args.api_key)
    
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
    else:
        device = f"cuda:{torch.cuda.current_device()}"
    model_name = get_model_name_from_path(args.model_path)
    if args.model_base is None:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path, 
            model_base=args.model_base, 
            model_name=model_name,
            load_8bit=args.load_8bit, 
            load_4bit=args.load_4bit, 
            device=device
        )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base, 
            model_name="llava_lora_model",
            load_8bit=args.load_8bit, 
            load_4bit=args.load_4bit, 
            device=device
        )

    conv_mode = "llava_v1"

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
    
    judgement = {}
    run_all = ['run1']
    for run in run_all:
        judgement[run] = {}
    _gram1, _gram2, _gram3, _gram4 = 0, 0, 0, 0
    
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
    
    for _data in tqdm.tqdm(val_images):
        image_id = _data["image_id"]
        image_path = id2path[int(image_id)]
        image = Image.open(image_path).convert("RGB")
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        conv = conv_templates[conv_mode].copy()
        if image is not None:
            inp = "Describe this image in detail."
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                num_beams=5,
                temperature=1.0,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")
        # get GPT judgement
        description = get_desc(id2img, id2reg, int(image_id))
        model_cap_sep, is_repeated = get_model_cap(outputs)
        # calculate repetition
        gram1 = cal_repetition(outputs,1)
        gram2 = cal_repetition(outputs,2)
        gram3 = cal_repetition(outputs,3)
        gram4 = cal_repetition(outputs,4)
        _gram1 += gram1
        _gram2 += gram2
        _gram3 += gram3
        _gram4 += gram4
            
        # skip gpt judgement 
        if args.no_gpt_judge:
            continue
            
        factual_text = ""
        if str(image_id) in factual_inf:
            for text in factual_inf[str(image_id)]:
                factual_text += text
                factual_text += "\n"
        # GPT judgement
        judge_prompt = GPT_JUDGE_PROMPT.format(description, factual_text, model_cap_sep)
        if len(judge_prompt) > 15000:
            print(f"skip {image_id} for too long prompt!")
            continue
        
        
        for run in run_all:
            while True:
                judge = get_gpt_response(prompt=judge_prompt)
                if "Judgement" not in judge:
                    print(f"No judgement found for {image_id}")
                    continue
                else:
                    break
            # post-process
            final_judge = post_process_no_revise(judge, outputs)
            judgement[run][image_id] = {
                "raw_judgement": judge,
                "model_response": outputs,
                "judgement": final_judge,
            }
        
    if args.no_gpt_judge:
        print(f"gram-1 repetition: {round(_gram1/len(val_images), 3)}")
        print(f"gram-2 repetition: {round(_gram2/len(val_images), 3)}")
        print(f"gram-3 repetition: {round(_gram3/len(val_images), 3)}")
        print(f"gram-4 repetition: {round(_gram4/len(val_images), 3)}")
    else:
        base_eval_path = "./ha_dpo/models/llava-v1_5/shr_eval_results"
        localtime = time.asctime( time.localtime(time.time()) ).replace(' ', '_')
        if not os.path.exists(os.path.join(base_eval_path)):
            os.mkdir(os.path.join(base_eval_path))
        # dump config file
        eval_path = os.path.join(os.path.join(base_eval_path, localtime))
        os.mkdir(eval_path)
        # save metrics
        metrics = {}
        for run in run_all:
            metrics[run] = {}
            get_metric(judgement[run], metrics[run])
        # repetition
        metrics['gram-1-repetition'] = round(_gram1/len(val_images), 3)
        metrics['gram-2-repetition'] = round(_gram2/len(val_images), 3)
        metrics['gram-3-repetition'] = round(_gram3/len(val_images), 3)
        metrics['gram-4-repetition'] = round(_gram4/len(val_images), 3)
        # halucination ratio
        metrics["mean_hal_ratio"] = round(
            sum(metrics[run]["hal_sents_ratio"] for run in run_all)/len(run_all), 3
        )
        metrics["model_base"] = args.model_base
        metrics["model_path"] = args.model_path
        # dump judgement file
        with open(os.path.join(base_eval_path, localtime, 'judgement.json'), "w") as f:
            json.dump(judgement, f)
        # dump metric file
        with open(os.path.join(base_eval_path, localtime, 'metrics.json'), "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--api-key", type=str, required=True, help="key to the OPENAI API.")
    parser.add_argument("--vg-path", type=str, required=True, help="path to vg file.")
    parser.add_argument("--shr-path", type=str, required=True, help="path to SHR annotation file.")
    parser.add_argument("--no-gpt-judge", default=False, action='store_true', help="whether not to do GPT evaluation. If True, only evaluate ngram repitition.")
    args = parser.parse_args()
    main(args)