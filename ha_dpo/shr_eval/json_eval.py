import os
import json
import tqdm
import argparse
from PIL import Image

from ha_dpo.shr_eval.shr_utils import *
from ha_dpo.shr_eval.gpt_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="SHR Evaluation")
    parser.add_argument("--api-key", type=str, required=True, help="key to the OPENAI API.")
    parser.add_argument("--json-file", type=str, required=True, help="path to the json file, where model responses are stored.")
    parser.add_argument("--vg-path", type=str, required=True, help="path to vg file.")
    parser.add_argument("--shr-path", type=str, required=True, help="path to SHR annotation.")
    parser.add_argument("--no-gpt-judge", default=False, action='store_true', help="whether not to do GPT evaluation. If True, only evaluate ngram repetition.")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    
    # setup openai
    setup_openai(args.api_key)
    
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
    
    # json file to be evaluated
    json_file = json.load(open(args.json_file))
    
    judgement = {}
    run_all = ['run1']
    for run in run_all:
        judgement[run] = {}
    
    _gram1, _gram2, _gram3, _gram4 = 0, 0, 0, 0
    for _data in tqdm.tqdm(val_images):
        image_id = _data["image_id"]
        # ask model to describe the image
        model_response = json_file[str(image_id)]
        
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
                "mode_response": model_response,
                "judgement": final_judge,
            }
        
    whole_sample_cnt = len(val_images)
    if args.no_gpt_judge:
        print(f"gram-1 repetition: {round(_gram1/whole_sample_cnt, 3)}")
        print(f"gram-2 repetition: {round(_gram2/whole_sample_cnt, 3)}")
        print(f"gram-3 repetition: {round(_gram3/whole_sample_cnt, 3)}")
        print(f"gram-4 repetition: {round(_gram4/whole_sample_cnt, 3)}")
    else:
        base_eval_path = "./ha_dpo/shr_eval/shr_eval_results"
        if not os.path.exists(base_eval_path):
            os.mkdir(base_eval_path)
        localtime = time.asctime( time.localtime(time.time()) ).replace(' ', '_')
        # dump config file
        eval_path = os.path.join(base_eval_path, localtime)
        os.mkdir(eval_path)
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