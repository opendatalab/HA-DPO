from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.cc_sbu import CCSBUDataset


@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        min_length = self.config.get("min_length", 1)

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
            length=self.config.get("length", 1),
            min_length=min_length
        )

        return datasets
