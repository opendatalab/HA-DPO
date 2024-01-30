# LLaVA-1.5

## Model Training

#### 1. model preparation

The LLaVA version we use in HA-DPO is **LLaVA-v1.5-7b**. Before training, prepare following:

- **Fine-tuned LLaVA model**. Download [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b).

```python
# download command
from huggingface_hub import snapshot_download
snapshot_download(repo_id="liuhaotian/llava-v1.5-7b")
```

If you download language model weights to a user-specified path using ```git lfs``` rather than huggingface provided download API (such as ```from_pretrained``` or ```snapshot_download```), replace all ```liuhaotian/llava-v1.5-7b``` to path of your downloaded model in training and evaluation. 

#### 2. data Preparation

follow instructions in [data preparation](ha_dpo/data/data_preparation.md) for LLaVA-1.5 data preparation.

#### 3. model training

We use LoRA adapters to fine-tune the language model of LLaVA-1.5 to train the model. Following training settings in LLaVA, all linear layers in the model are set to trainable. 8 A100 GPUs are used during fine-tuning.

<details>
<summary> Training command </summary>

```
deepspeed ha_dpo/models/llava-v1_5/train_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
    --deepspeed ha_dpo/models/llava-v1_5/scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --vg_path ha_dpo/data/VG \
    --desc_data_path ha_dpo/data/hadpo/llava-v1.5/desc_data.json \
    --pope_data_path ha_dpo/data/hadpo/llava-v1.5/pope_data.json \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ha_dpo/models/llava-v1_5/checkpoints/{model_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "llava-v1.5" \
    --beta 0.1
```
    
</details>

default parameters are as follows:

| epochs | learning rate | lora_r | lora_alpha | beta |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 2e-6 | 128 | 256 | 0.1 |


## Model Evaluation

### SHR Evaluation

Run the following command to evaluate on SHR:

```
python ha_dpo/models/llava-v1_5/shr_eval.py \
--api-key {openai_apikey} \
--vg-path ha_dpo/data/VG \
--shr-path ha_dpo/data/shr \
--model-base liuhaotian/llava-v1.5-7b \
--model-path ha_dpo/models/llava-v1_5/checkpoints/{model_name}
```

1. ```--api-key```: SHR evaluation relies on GPT-4. Provide the openai key, begin with ```sk```.
2. ```--model-path```: path to the the trained adapter weights.

After evaluation is finished, results are saved in ```ha_dpo/models/llava-v1_5/shr_eval_results/{localtime}/metrics.json```.

1. ```judgement.json```: detailed judgements in SHR evaluation.
2. ```metrics.json```: detailed metrics in SHR evaluation. ```mean_hal_ratio``` indicates the ration of hallucinated sentences, which is the main SHR result.

to reproduce results, use [trained adapter weights](https://huggingface.co/juliozhao/hadpo-llava-1.5) and set ```--model-path juliozhao/hadpo-llava-1.5```.


<details>
<summary> SHR results </summary>

| Model | HA-DPO | SHR |
|:--:|:--:|:--:|
| LLaVA-1.5 | :heavy_multiplication_x: | 36.7 |
| LLaVA-1.5 | :heavy_check_mark: | 34.0 |

</details>
    
### POPE Evaluation

**_step 1._** Firstly, inference answers using:

```
torchrun --nproc_per_node {NGPUS} --master_port $RANDOM ha_dpo/models/llava-v1_5/pope_eval.py \
    --coco_path ha_dpo/data/coco2014 \
    --pope_path ha_dpo/data/POPE \
    --model-path ha_dpo/models/llava-v1_5/checkpoints/{model_name} \
    --model-base liuhaotian/llava-v1.5-7b \
    --set {random/popular/adv}
```

1. ```--set```: validation sets in POPE, choose ```random/popular/adv```. After inference, the answer file will be generated under the folder of LLaVA.
2. ```--model-path```: path to the the trained adapter weights.

to reproduce results, use [trained adapter weights](https://huggingface.co/juliozhao/hadpo-llava-1.5) and set ```--model-path juliozhao/hadpo-llava-1.5```.

**_step 2._** Set the path of answer file and the label file in ```ha_dpo/data/POPE/evaluate.py```.

Set ```ans_file``` to the path of the generated answer file in step 1, and set ```label_file``` to the path of label files under ```ha_dpo/data/POPE/output/coco```.

**_step 3._** Evaluate.

run ```python ha_dpo/data/POPE/evaluate.py``` to get results.

<details>
<summary> POPE results </summary>

**POPE Random**

| Model | HA-DPO | Accuracy | Precision | Recall | F1 Score | Yes Ratio (%) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| LLaVA-1.5 | :heavy_multiplication_x: | 89.60 | 88.77 | 90.66 | 89.70 | 51.06 |
| LLaVA-1.5 | :heavy_check_mark: | 90.53 | 92.99 | 87.66 | 90.25 | 47.13 |

**POPE Popular**

| Model | HA-DPO | Accuracy | Precision | Recall | F1 Score | Yes Ratio (%) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| LLaVA-1.5 | :heavy_multiplication_x: | 86.20 | 83.23 | 90.66 | 86.79 | 54.46 |
| LLaVA-1.5 | :heavy_check_mark: | 87.90 | 88.07 | 87.66 | 87.81 | 49.76 |

**POPE Adversarial**

| Model | HA-DPO | Accuracy | Precision | Recall | F1 Score | Yes Ratio (%) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| LLaVA-1.5 | :heavy_multiplication_x: | 79.76 | 74.43 | 90.66 | 81.75 | 60.90 |
| LLaVA-1.5 | :heavy_check_mark: | 81.46 | 77.99 | 87.66 | 82.54 | 56.20 |

</details>
    
> :warning: NOTICE:
> 1. The optimal parameters can differ according to the environment of your machine, you can adjust these parameters according to the behavior of LVLM.
> 2. For baseline LLaVA-1.5 results, don't set ```--model-base``` and set ```--model-path``` to ```liuhaotian/llava-v1.5-7b``` in the evaluation command.