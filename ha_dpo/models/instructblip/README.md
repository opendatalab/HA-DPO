# InstructBLIP

## Model Training

#### 1. model preparation

The InstructBLIP version we use in HA-DPO is **InstructBLIP-13b** (based on **vicuna-13b-v1.1**). Before training, prepare following:

- **Language Model**. Download [vicuna-13b-v1.1](https://huggingface.co/lmsys/vicuna-13b-v1.1)

```python
# download command
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lmsys/vicuna-13b-v1.1")
```

If you download language model weights to a user-specified path using ```git lfs``` rather than huggingface provided download API (such as ```from_pretrained``` or ```snapshot_download```), there are few things you should notice:

1. specify the model path of ```llm_model``` in ```ha_dpo/models/instructblip/vigc/projects/ha-dpo/instruct_vicuna13b.yaml```.

2. when passing path of language model weights in training or evaluation commands, use path to your downloaded model instead of ```lmsys/vicuna-13b-v1.1```.

- **Pre-trained InstructBLIP-13B weight**. Download pretrained InstructBLIP-13B checkpoint [instruct_blip_vicuna13b_trimmed.pth](https://huggingface.co/juliozhao/instruct_blip_vicuna13b_trimmed/tree/main). Put downloaded checkpoint under ```ha_dpo/models/instructblip```. (this model weights are used under [LICENSE](https://github.com/salesforce/LAVIS/blob/main/LICENSE.txt))

#### 2. Data Preparation

follow instructions in [data preparation](ha_dpo/data/data_preparation.md) for InstructBLIP data preparation.

#### 3. config accelerate

run ```accelerate config```, default we use:

```
gpus = 8
bf16 = True
```

#### 4. model training

We use LoRA adapters to fine-tune the language model of InstructBLIP to train the model. Trainable parameters are ```["q_proj","k_proj","v_proj"]```. 8 A100 GPUs are used during fine-tuning.

<details>
<summary> Training command: </summary>

```
accelerate launch --main_process_port $RANDOM ha_dpo/models/instructblip/train_dpo.py \
--lora_r 64 \
--cfg_path ha_dpo/models/instructblip/vigc/projects/ha-dpo/instruct_vicuna13b.yaml \
--pope_train_data_path ha_dpo/data/hadpo/instructblip/pope_data.json \
--desc_train_data_path ha_dpo/data/hadpo/instructblip/desc_data.json \
--vg_path ha_dpo/data/VG \
--gradient_checkpointing False \
--num_train_epoch 1 \
--run_name "instructblip" \
--gradient_accumulation_steps 4 \
--learning_rate 4e-6 \
--warmup_steps 0 \
--per_device_train_batch_size 1 \
--output_dir 'ha_dpo/models/instructblip/vigc/output/{model_name}' \
--logging_steps 4
```
    
</details>

default parameters are as follows:

| epoch | learning rate | lora_r | lora_alpha | beta |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 4e-6 | 64 | 16 | 0.1 |

#### 5. merge LoRA adapters

Merge LoRA adapters into language model:

```
python ha_dpo/models/instructblip/merge_peft_adapter.py \
--adapter_model_name ha_dpo/models/instructblip/vigc/output/{model_name} \
--base_model_name lmsys/vicuna-13b-v1.1 \
--output_name {path_to_merged_llm}
```

1. ```--adapter_model_name```: path to the saved adapter weights during training.
2. ```--output_name```: path where the merged language model weights are saved.

to reproduce results, use [trained adapter weights](https://huggingface.co/juliozhao/hadpo-instructblip) and set ```--adapter_model_name juliozhao/hadpo-instructblip```.

## Model Evaluation

### SHR Evaluation

Run the following command to evaluate on SHR:

```
python ha_dpo/models/instructblip/shr_eval.py \
--api-key {openai_apikey} \
--cfg-path ha_dpo/models/instructblip/vigc/projects/ha-dpo/instruct_vicuna13b.yaml \
--llm-model {path_to_merged_llm} \
--vg-path ha_dpo/data/VG \
--shr-path ha_dpo/data/shr
```

1. ```--api-key```: SHR evaluation relies on GPT-4. Provide the openai key, begin with ```sk```.
2. ```--llm-model```: path to the the merge language model weight.

After evaluation is finished, results are saved in ```ha_dpo/models/instructblip/shr_eval_results/{localtime}/metrics.json```.

1. ```judgement.json```: detailed judgements in SHR evaluation.
2. ```metrics.json```: detailed metrics in SHR evaluation. ```mean_hal_ratio``` indicates the ration of hallucinated sentences, which is the main SHR result.

<details>
<summary> SHR results </summary>

| Model | HA-DPO | SHR |
|:--:|:--:|:--:|
| InstructBLIP-13B | :heavy_multiplication_x: | 51.2 |
| InstructBLIP-13B | :heavy_check_mark: | 49.1 |

</details>
    
### POPE Evaluation

**_step 1._** Firstly, inference answers using:

```
torchrun --nproc_per_node {NGPUs} ha_dpo/models/instructblip/pope_eval.py \
--cfg-path ha_dpo/models/instructblip/vigc/projects/ha-dpo/instruct_vicuna13b.yaml \
--llm-model {path_to_merged_llm} \
--set {random/popular/adv} \
--pope-path ha_dpo/data/POPE \
--coco-path ha_dpo/data/coco2014
```

1. ```--set```: validation sets in POPE, choose ```random/popular/adv```. After inference, the answer file will be generated under the folder of InstructBLIP.
2. ```--model-path```: path to the the merged language model weights.

**_step 2._** Set the answer file and the label file in ```ha_dpo/data/POPE/evaluate.py```.

Set ```ans_file``` to the path of the generated answer file in step 1, and set ```label_file``` to the path of label files under ```ha_dpo/data/POPE/output/coco```.

**_step 3._** Evaluate.

run ```python ha_dpo/data/POPE/evaluate.py``` to get results.

<details>
<summary> POPE results </summary>

**POPE Random**

| Model | HA-DPO | Accuracy | Precision | Recall | F1 Score | Yes Ratio (%) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| InstructBLIP-13B | :heavy_multiplication_x: | 88.70 | 85.03 | 93.93 | 89.26 | 55.23 |
| InstructBLIP-13B | :heavy_check_mark: | 89.83 | 93.07 | 86.06 | 89.43 | 46.23 |

**POPE Popular**

| Model | HA-DPO | Accuracy | Precision | Recall | F1 Score | Yes Ratio (%) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| InstructBLIP-13B | :heavy_multiplication_x: | 81.36 | 75.06 | 93.93 | 83.44 | 62.56 |
| InstructBLIP-13B | :heavy_check_mark: | 85.76 | 85.55 | 86.06 | 85.80 | 50.03 |

**POPE Adversarial**

| Model | HA-DPO | Accuracy | Precision | Recall | F1 Score | Yes Ratio (%) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| InstructBLIP-13B | :heavy_multiplication_x: | 74.50 | 67.64 | 93.93 | 78.64 | 69.43 |
| InstructBLIP-13B | :heavy_check_mark: | 80.70 | 77.72 | 86.06 | 81.68 | 55.36 |

</details>
    
> :warning: NOTICE:
> 1. The optimal parameters can differ according to the environment of your machine, you can adjust these parameters according to the behavior of LVLM.
> 2. For baseline model results, don't set ```--llm-model``` in the evaluation command.