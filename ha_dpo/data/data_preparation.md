# Data Preparation

### Dataset card

Prepare different data according to the following dataset card. 

| **Stage** | **Data Name** | **Data Path** | 
| :--: | :--: | :--: |
| MiniGPT-4 DPO Training | Visual Genome, HA-DPO data, CCSBUAlign | ```ha_dpo/data/VG```, ```ha_dpo/data/hadpo/minigpt4```, ```ha_dpo/data/cc_sbu_align``` |
| LLaVA-1.5 DPO Training | Visual Genome, HA-DPO data | ```ha_dpo/data/VG```, ```ha_dpo/data/ha_dpo/llava-v1.5``` |
| InstructBLIP DPO Training | Visual Genome, HA-DPO data | ```ha_dpo/data/VG```, ```ha_dpo/data/ha_dpo/InstructBLIP``` |
| SHR Evaluation | Visual Genome, SHR annotation | ```ha_dpo/data/VG``` | ```ha_dpo/data/cc_sbu_align``` |
| POPE Evaluation | COCO val2014, POPE annotation | ```ha_dpo/data/coco2014```, ```ha_dpo/data/POPE``` |

For example, if you want to perform MiniGPT-4 model training, following data are needed:

1. MiniGPT-4 hallucination-aware positive-negative data 

2. Visual Genome

3. CCSBUAlign

### Hallucination-aware positive-negative data

We provide hallucination-aware positive-negative data targeted at each LVLM. Download data from the following and put data under ```ha_dpo/data/hadpo```:

| MiniGPT-4 | LLaVA-1.5 | InstructBLIP |
| :--: | :--: | :--: |
| [huggingface](https://huggingface.co/datasets/juliozhao/hadpo-data/tree/main/hadpo/minigpt4), [opendatalab](https://openxlab.org.cn/datasets/zzy8782180/HA-DPO/tree/main/hadpo/minigpt4) | [huggingface](https://huggingface.co/datasets/juliozhao/hadpo-data/tree/main/hadpo/llava-v1.5), [opendatalab](https://openxlab.org.cn/datasets/zzy8782180/HA-DPO/tree/main/hadpo/llava-v1.5) | [huggingface](https://huggingface.co/datasets/juliozhao/hadpo-data/tree/main/hadpo/instructblip), [opendatalab](https://openxlab.org.cn/datasets/zzy8782180/HA-DPO/tree/main/hadpo/instructblip) |

<details>
<summary> data structure </summary>

```
ha_dpo/data/hadpo
├── llava-v1.5
│   ├── desc_data.json
│   └── pope_data.json
├── InstructBLIP
│   ├── desc_data.json
│   └── pope_data.json
└── minigpt4
    ├── desc_data.json
    └── pope_data.json
```
    
</details>

### Visual Genome

Download these data from [Visual Genome v2](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) and put data under ```ha_dpo/data/VG```:

1. images (both part1 and part2)
2. image meta-data
3. region descriptions

<details>
<summary> data structure </summary>

```
ha_dpo/data/VG
├── image_data.json
├── region_descriptions.json
├── VG_100K
│    └──...
└── VG_100K_2
     └──...
```

</details>
    
### SHR annotation

Download SHR human-annotated factual annotation from [download](https://huggingface.co/datasets/juliozhao/SHR/tree/main) and put data under ```ha_dpo/data/shr```. 

<details>
<summary> data structure </summary>

```
ha_dpo/data/shr
├── shr_factual_part1.jsonl
├── shr_factual_part2.jsonl
└── val_images_final.json
```
    
</details>

### POPE

Clone POPE repo from [official website](https://github.com/RUCAIBox/POPE) and put data under ```ha_dpo/data```.
    
<details>
<summary> data structure </summary>

```
ha_dpo/data/
└── POPE
     └── ...
```
    
</details>

### COCO2014 val images

For POPE evaluation, coco2014 images are required. Download COCO2014 validation images from [COCO](https://cocodataset.org/#download) and put data under ```ha_dpo/data/shr```. 

<details>
<summary> data structure </summary>

```
ha_dpo/data/coco2014
└── val2014
     └── ...
```

</details>

### CCSBUAlign

CCSBUAlign is the SFT data used in MiniGPT-4. We use this data **only during MiniGPT-4 model training** to ensure the stability of preference learning. Refer to [official website](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/dataset/README_2_STAGE.md) to obtain CCSBUAlign data. Put data under ```ha_dpo/data/cc_sbu_align```.

<details>
<summary> data structure </summary>

```
ha_dpo/data/cc_sbu_align
├── filter_cap.json
└── image
     └── ...
```
    
</details>