import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.vig_eval import A_OKVQA_VIG_EvalDataset, COCO2017_Eval_Dataset, OKVQA_VIG_EvalDataset, \
    VQAv2_VIG_EvalDataset, LlavaEvalDataset, COCO2017_Given_Question_Eval_Dataset, Object365_Eval_Dataset


@registry.register_builder("instruct_blip_aokvqa_vig_eval")
class AOKVQAEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = A_OKVQA_VIG_EvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/a-okvqa/vig_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building A-OKVQA VIG Eval datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_given_q_coco2017_vig_test")
class COCO_Jiahui_VQGBuilder(BaseDatasetBuilder):
    eval_dataset_cls = COCO2017_Given_Question_Eval_Dataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco_pseudo/test_given_q_vig.yaml"
    }

    def build_datasets(self):
        logging.info("Building COCO2017 Given Question VIG Test datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = self.config.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_coco2017_vig_test")
class COCOPseudoEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = COCO2017_Eval_Dataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco_pseudo/vig_test.yaml"
    }

    def build_datasets(self):
        logging.info("Building COCO2017 VIG Test datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        filter_dataset = self.config.get("filter", [])
        anno_path = build_info.annotation,
        image_id_path = build_info.get("image_ids", None)
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
            image_ids_file=image_id_path,
            filter=filter_dataset
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_object365_vig_test")
class COCOPseudoEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = Object365_Eval_Dataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco_pseudo/vig_test_object365.yaml"
    }

    def build_datasets(self):
        logging.info("Building Object365 VIG Test datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        filter_dataset = self.config.get("filter", [])
        anno_path = build_info.annotation,
        image_id_path = build_info.get("image_ids", None)
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
            image_ids_file=image_id_path,
            filter=filter_dataset
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_okvqa_vig_eval")
class OKVQAEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = OKVQA_VIG_EvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/vig_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building OKVQA VIG Eval datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_vqav2_vig_eval")
class VQAv2EvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = VQAv2_VIG_EvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqav2/vig_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building VQAv2 VIG Eval datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_llava_vig_eval")
class LlavaVQGAEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = LlavaEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava_instruct150k/vig_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building LLava VIG Eval datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
        )
        _ = datasets['eval'][0]

        return datasets
