import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from vigc.datasets.datasets.vigc_train import LlavaCompDataset, LlavaDescDataset, LlavaConvDataset, A_OKVQADataset, \
    OKVQADataset, VQAv2Dataset
from vigc.datasets.datasets.coco_pseudo import COCO_Pseudo_Dataset

TRAIN_DATASET_DICT = {
    "llava_comp": LlavaCompDataset,
    "llava_desc": LlavaDescDataset,
    "llava_conv": LlavaConvDataset,
    "a_okvqa": A_OKVQADataset,
    "okvqa": OKVQADataset,
    "vqav2": VQAv2Dataset,
    "coco_pseudo": COCO_Pseudo_Dataset
}

ALL_DATASET_CONFIG_DICT = {
    "llava_comp": "configs/datasets/llava_instruct150k/{task}/trainval_llava_comp.yaml",
    "llava_desc": "configs/datasets/llava_instruct150k/{task}/trainval_llava_desc.yaml",
    "llava_conv": "configs/datasets/llava_instruct150k/{task}/trainval_llava_conv.yaml",
    "a_okvqa": "configs/datasets/a-okvqa/{task}/train.yaml",
    "okvqa": "configs/datasets/okvqa/{task}/train.yaml",
    "vqav2": "configs/datasets/vqav2/{task}/train.yaml",
    "coco_pseudo": "configs/datasets/coco_pseudo/{task}/train.yaml",
}

TASK2BULDER = {
    "llava_comp": "instruct_blip_llava_comp_{task}",
    "llava_desc": "instruct_blip_llava_desc_{task}",
    "llava_conv": "instruct_blip_llava_conv_{task}",
    "a_okvqa": "instruct_blip_aokvqa_{task}",
    "okvqa": "instruct_blip_okvqa_{task}",
    "vqav2": "instruct_blip_vqav2_{task}",
    "coco_pseudo": "instruct_blip_coco_pseudo_{task}"
}


class VIGCBuilder(BaseDatasetBuilder):
    TASK = "vig"
    TYPE = None
    DATASET_CONFIG_DICT = {"default": None}

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.

        self.build_processors()

        task = self.TASK
        type_ = self.TYPE
        logging.info(f"Building {type_} {task} Training datasets...")

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = TRAIN_DATASET_DICT[type_]
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=anno_path,
            task=task
        )
        _ = datasets['train'][0]

        return datasets


############################ VQGA  ##############################

@registry.register_builder("instruct_blip_llava_comp_vig")
class LlavaCompVIGBuilder(VIGCBuilder):
    TYPE = "llava_comp"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task="vig")}


@registry.register_builder("instruct_blip_llava_desc_vig")
class LlavaDescVIGBuilder(VIGCBuilder):
    TYPE = "llava_desc"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task="vig")}


@registry.register_builder("instruct_blip_llava_conv_vig")
class LlavaConvVIGBuilder(VIGCBuilder):
    TYPE = "llava_conv"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task="vig")}


@registry.register_builder("instruct_blip_aokvqa_vig")
class A_OKVQA_VIGBuilder(VIGCBuilder):
    TYPE = "a_okvqa"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task="vig")}


@registry.register_builder("instruct_blip_okvqa_vig")
class OKVQA_VIGBuilder(VIGCBuilder):
    TYPE = "okvqa"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task="vig")}


@registry.register_builder("instruct_blip_vqav2_vig")
class VQAv2_VIGBuilder(VIGCBuilder):
    TYPE = "vqav2"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task="vig")}


@registry.register_builder("instruct_blip_coco_pseudo_vig")
class COCO_Pseudo_VIGBuilder(VIGCBuilder):
    TYPE = "coco_pseudo"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task="vig")}

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.

        self.build_processors()

        task = self.TASK
        type_ = self.TYPE
        logging.info(f"Building {type_} {task} Training datasets...")

        build_info = self.config.build_info
        filter_dataset = self.config.get("filter", [])
        image_id_path = build_info.image_ids
        anno_path = self.config.annotation,
        vis_root = build_info.images
        topk = self.config.get("topk", None)
        score_thr = self.config.get("score_thr", 0.)

        datasets = dict()

        # create datasets
        dataset_cls = TRAIN_DATASET_DICT[type_]
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=anno_path,
            task=task,
            image_ids_file=image_id_path,
            filter=filter_dataset,
            topk=topk,
            threshold=score_thr
        )
        _ = datasets['train'][0]

        return datasets


################################  VQA  ######################################
@registry.register_builder("instruct_blip_llava_comp_vic")
class LlavaCompVQABuilder(VIGCBuilder):
    TASK = "vic"
    TYPE = "llava_comp"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_llava_desc_vic")
class LlavaDescVQABuilder(VIGCBuilder):
    TASK = "vic"
    TYPE = "llava_desc"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_llava_conv_vic")
class LlavaConvVQABuilder(VIGCBuilder):
    TASK = "vic"
    TYPE = "llava_conv"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_aokvqa_vic")
class A_OKVQA_VQABuilder(VIGCBuilder):
    TYPE = "a_okvqa"
    TASK = "vic"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_okvqa_vic")
class OKVQA_VQABuilder(VIGCBuilder):
    TYPE = "okvqa"
    TASK = "vic"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_vqav2_vic")
class VQAv2_VQABuilder(VIGCBuilder):
    TYPE = "vqav2"
    TASK = "vic"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_coco_pseudo_vic")
class COCO_Pseudo_VQABuilder(COCO_Pseudo_VIGBuilder):
    TYPE = "coco_pseudo"
    TASK = "vic"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


################################  VQG  ######################################
@registry.register_builder("instruct_blip_llava_comp_vqg")
class LlavaCompVQGBuilder(VIGCBuilder):
    TASK = "vqg"
    TYPE = "llava_comp"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_llava_desc_vqg")
class LlavaDescVQGBuilder(VIGCBuilder):
    TASK = "vqg"
    TYPE = "llava_desc"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_llava_conv_vqg")
class LlavaConvVQGBuilder(VIGCBuilder):
    TASK = "vqg"
    TYPE = "llava_conv"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_aokvqa_vqg")
class A_OKVQA_VQGBuilder(VIGCBuilder):
    TYPE = "a_okvqa"
    TASK = "vqg"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_okvqa_vqg")
class OKVQA_VQGBuilder(VIGCBuilder):
    TYPE = "okvqa"
    TASK = "vqg"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_vqav2_vqg")
class VQAv2_VQGBuilder(VIGCBuilder):
    TYPE = "vqav2"
    TASK = "vqg"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}


@registry.register_builder("instruct_blip_coco_pseudo_vqg")
class COCO_Pseudo_VQGBuilder(COCO_Pseudo_VIGBuilder):
    TYPE = "coco_pseudo"
    TASK = "vqg"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task=TASK)}
