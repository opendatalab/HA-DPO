import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.vic_eval import A_OKVQAEvalDataset, VQAv2EvalDataset, A_OKVQATestDataset


@registry.register_builder("instruct_blip_aokvqa_eval")
class AOKVQAEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = A_OKVQAEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/a-okvqa/eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building A-OKVQA Eval datasets ...")
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


@registry.register_builder("instruct_blip_aokvqa_test")
class AOKVQATestBuilder(BaseDatasetBuilder):
    eval_dataset_cls = A_OKVQATestDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/a-okvqa/test.yaml"
    }

    def build_datasets(self):
        logging.info("Building A-OKVQA Test datasets ...")
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


@registry.register_builder("instruct_blip_vqav2_eval")
class VQAv2EvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = VQAv2EvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqav2/eval.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building VQAv2 Eval datasets...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        sample_ratio = 20

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path,
            sample_ratio=sample_ratio
        )
        _ = datasets['eval'][0]

        return datasets


@registry.register_builder("instruct_blip_okvqa_eval")
class OKVQAEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = VQAv2EvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/eval.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building OKVQA Eval datasets...")
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
            anno_file=anno_path
        )
        _ = datasets['eval'][0]

        return datasets
