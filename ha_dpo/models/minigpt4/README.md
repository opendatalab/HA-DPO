# MiniGPT-4

## Model Training

#### 1. model preparation

The MiniGPT-4 version we use in HA-DPO is **MiniGPT-4-llama2-7b** (based on **llama-2-7b-chat-hf**). Before training, prepare following:

- **Language Model**. Go to llama2 [official site](https://ai.meta.com/llama/) to request for model weights and download huggingface format [llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

```python
# download command
from huggingface_hub import snapshot_download
snapshot_download(repo_id="meta-llama/Llama-2-7b-chat-hf")
```

If you download language model weights to a user-specified path using ```git lfs``` rather than huggingface provided download API (such as ```from_pretrained``` or ```snapshot_download```), there are few things you should notice:

1. specify the model path of ```llama_model``` in ```ha_dpo/models/minigpt4/minigpt4/configs/models/minigpt4_llama2.yaml```.

2. when passing path of language model weights in training or evaluation commands, use path to your downloaded model instead of ```meta-llama/Llama-2-7b-chat-hf```.

- **Instruction fine-tunned MiniGPT-4-llama2-7b weight**. Download stage-2 instruction fine-tunned MiniGPT-4-llama2-7b weight [pretrained_minigpt4_llama2_7b.pth](https://huggingface.co/juliozhao/pretrained_minigpt4_llama2_7b/tree/main). Put downloaded checkpoint under ```ha_dpo/models/minigpt4```. (this model weights are used under [LICENSE](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/LICENSE.md))

#### 2. data Preparation

follow instructions in [data preparation](ha_dpo/data/data_preparation.md) for MiniGPT-4 data preparation.

#### 3. config accelerate

run ```accelerate config```, default we use:

```
gpus = 8
bf16 = True
```

#### 4. model training

We use LoRA adapters to fine-tune the language model of MiniGPT-4 to train the model. Trainable parameters are ```["q_proj","k_proj","v_proj"]```. 8 A100 GPUs are used during fine-tuning.

<details>
<summary> Training command </summary>

```
accelerate launch --main_process_port $RANDOM ha_dpo/models/minigpt4/train_dpo.py \
--cfg_path ha_dpo/models/minigpt4/train_configs/minigpt4_llama2_stage3_dpo.yaml \
--auxilary True \
--ccsbualign_data_path ha_dpo/data/cc_sbu_align \
--pope_train_data_path ha_dpo/data/hadpo/minigpt4/pope_data.json \
--desc_train_data_path ha_dpo/data/hadpo/minigpt4/desc_data.json \
--vg_path ha_dpo/data/VG \
--lora_r 64 \
--gradient_checkpointing False \
--per_device_train_batch_size 1 \
--learning_rate 1e-4 \
--beta 0.1 \
--gamma 0.5 \
--gradient_accumulation_steps 4 \
--max_steps 1000 \
--output_dir 'ha_dpo/models/minigpt4/minigpt4/output/{model_name}' \
--logging_steps 4
```
    
</details>

default parameters are as follows:

| steps | learning rate | lora_r | lora_alpha | beta | gamma |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 1k | 1e-4 | 64 | 16 | 0.1 | 0.5 |

#### 5. merge LoRA adapters

Merge LoRA adapters into language model:

```
python ha_dpo/models/minigpt4/merge_peft_adapter.py \
--adapter_model_name ha_dpo/models/minigpt4/minigpt4/output/{model_name} \
--base_model_name meta-llama/Llama-2-7b-chat-hf \
--output_name {path_to_merged_llm}
```

1. ```--adapter_model_name```: path to the saved adapter weights during training.
2. ```--output_name```: path where the merged language model weights are saved.

to reproduce results, use [trained adapter weights](https://huggingface.co/juliozhao/hadpo-minigpt4) and set ```--adapter_model_name juliozhao/hadpo-minigpt4```.

## Model Evaluation

### SHR Evaluation

Run the following command to evaluate on SHR:

```
python ha_dpo/models/minigpt4/shr_eval.py \
--api-key {openai_apikey} \
--cfg-path ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml \
--llama-model {path_to_merged_llm} \
--vg-path ha_dpo/data/VG \
--shr-path ha_dpo/data/shr
```

1. ```--api-key```: SHR evaluation relies on GPT-4. Provide the openai key, begin with ```sk```.
2. ```--llama-model```: path to the the merge language model weight.

After evaluation is finished, results are saved in ```ha_dpo/models/minigpt4/shr_eval_results/{localtime}/metrics.json```.

1. ```judgement.json```: detailed judgements in SHR evaluation.
2. ```metrics.json```: detailed metrics in SHR evaluation. ```mean_hal_ratio``` indicates the ration of hallucinated sentences, which is the main SHR result.


<details>
<summary> SHR results </summary>

| Model | HA-DPO | SHR |
|:--:|:--:|:--:|
| MiniGPT-4-Llama2-7B | :heavy_multiplication_x: | 47.3 |
| MiniGPT-4-Llama2-7B | :heavy_check_mark: | 44.4 |

</details>
    
### POPE Evaluation

**_step 1._** Firstly, inference answers using:

```
torchrun --nproc-per-node {NGPUs} --master-port $RANDOM ha_dpo/models/minigpt4/pope_eval.py \
--set {random/popular/adv} \
--cfg-path ha_dpo/models/minigpt4/eval_configs/minigpt4_llama2_eval.yaml \
--llama-model {path_to_merged_llm} \
--pope-path ha_dpo/data/POPE \
--coco-path ha_dpo/data/coco2014
```

1. ```--set```: validation sets in POPE, choose between ```random/popular/adv```. After inference, the answer file will be generated under the folder of LLaVA.
2. ```--llama-model```: path to the the merged language model weights.

**_step 2._** Set the answer file and the label file in ```ha_dpo/data/POPE/evaluate.py```.

Set ```ans_file``` to the path of the generated answer file in step 1, and set ```label_file``` to the path of label files under ```ha_dpo/data/POPE/output/coco```.

**_step 3._** Evaluate.

run ```python ha_dpo/data/POPE/evaluate.py``` to get results.

<details>
<summary> POPE results </summary>

**POPE Random**

| Model | HA-DPO | Accuracy | Precision | Recall | F1 Score | Yes Ratio (%) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| MiniGPT-4-Llama2-7B | :heavy_multiplication_x: | 51.13    | 50.57 | 99.80 | 67.13 | 98.66 |
| MiniGPT-4-Llama2-7B | :heavy_check_mark: | 86.13 | 92.81 | 78.33 | 84.96 | 42.20 |

**POPE Popular**

| Model | HA-DPO | Accuracy | Precision | Recall | F1 Score | Yes Ratio (%) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| MiniGPT-4-Llama2-7B | :heavy_multiplication_x: | 51.46    | 50.74 | 99.53 | 67.72 | 98.06 |
| MiniGPT-4-Llama2-7B | :heavy_check_mark: | 79.50 | 80.20 | 78.33 | 79.25 | 48.83 |

**POPE Adversarial**

| Model | HA-DPO | Accuracy | Precision | Recall | F1 Score | Yes Ratio (%) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| MiniGPT-4-Llama2-7B | :heavy_multiplication_x: | 51.26 | 50.64 | 99.66 | 67.16 | 98.40 |
| MiniGPT-4-Llama2-7B | :heavy_check_mark: | 75.66 | 74.36 | 78.33 | 76.29 | 52.66 |

</details>
    
> :warning: NOTICE:
> 1. Because of the instability in MiniGPT-4 DPO training, we follow [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) and add an auxiliary language modeling task using its original SFT data (--auxilary True). This option can ensure the stability of model performance.
> 2. The optimal parameters can differ according to the environment of your machine, you can adjust these parameters according to the behavior of LVLM.
> 3. For baseline model results, don't set ```--llama-model``` in the evaluation command.