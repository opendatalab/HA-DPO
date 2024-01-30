import os
import tqdm
import json
import random
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from minigpt4.common.config import Config
from minigpt4.common.registry import registry

class PopeDataset(Dataset):
    """
        PopeDataset:
        use GPT-4 reconstructed POPE-format dataset.
    """
    def __init__(self, 
        data_path: str,
        vg_path: str,
        cfg: OmegaConf,
        auxilary_dataset = None,
    ):
        super(PopeDataset, self).__init__()
        self.data_path = data_path
        self.vg_path = vg_path
        self.cfg = cfg
        self.auxilary_dataset = auxilary_dataset
        
        # pos&neg data
        vis_cfg = self.cfg.datasets.cc_sbu_align.vis_processor.train
        vis_cfg.name = 'blip2_image_eval'
        self.vis_processor = self._build_proc_from_cfg(vis_cfg)
        self.data = json.load(open(self.data_path))
        
        # visual genome
        self.vg_image_data = json.load(open(os.path.join(self.vg_path, "image_data.json")))
        self.id2path = {
            _data["image_id"]:os.path.join(self.vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
            for _data in self.vg_image_data
        }
        self.load_data()
        
        if self.auxilary_dataset is not None:
            if "LOCAL_RANK" not in os.environ:
                self.aux_start, self.aux_end = 0, len(self.auxilary_dataset) - 1
            else:
                step = len(self.auxilary_dataset)//int(os.environ["WORLD_SIZE"]) + 1
                local_rank = int(os.environ["LOCAL_RANK"])
                self.aux_start = step * local_rank
                self.aux_end = min(step * (local_rank + 1), len(self.auxilary_dataset))
            self.aux_idx_generator = (idx for idx in range(self.aux_start, self.aux_end))
        
    def _build_proc_from_cfg(self, cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )
        
    def load_data(self):
        self.data_list = []
        print('Load POPE Data...')
        for anno in tqdm.tqdm(self.data):
            image_id = anno["image_id"]
            image_path = self.id2path[int(image_id)]
            self.data_list.append({
                "image_id": image_id,
                "image_path": image_path,
                "image": self.vis_processor(Image.open(image_path).convert("RGB")),
                "chosen": anno["chosen"],
                "rejected": anno["reject"],
                "data_type": "pope",
                "prompt": anno["question"],
            })
                
        print(f"Loaded {len(self.data_list)} pope data")
        
        print("Data Example:")
        print(self.data_list[0]["chosen"])
        print(self.data_list[0]["rejected"])
        
    def select(self, indices):
        return [
            self.__getitem__(idx) for idx in indices
        ]
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, index):
        sample = self.data_list[index]
        if self.auxilary_dataset is not None:
            try:
                aux_idx = next(self.aux_idx_generator)
            except BaseException:
                self.aux_idx_generator = (idx for idx in range(self.aux_start, self.aux_end))
                aux_idx = next(self.aux_idx_generator)
            aux_sample = self.auxilary_dataset[aux_idx]
            return {
                k: [sample[k], aux_sample[k]] for k in sample.keys()
            }
        else:
            return {
                k: [sample[k]] for k in sample.keys()
            }
        
        
class AugmentedCaptionDataset(Dataset):
    """
        AugmentedCaptionDataset:
        use GPT-3.5 augmented revised descriptions as chosen and augmented model response as rejected
    """
    def __init__(self, 
        data_path: str,
        vg_path: str,
        cfg: OmegaConf,
        seed: int,
        sample_strategy = "offline",
        auxilary_dataset = None,         
    ):
        super(AugmentedCaptionDataset, self).__init__()
        
        random.seed(seed)
        self.data_path = data_path
        self.vg_path = vg_path
        self.cfg = cfg
        self.auxilary_dataset = auxilary_dataset
        
        # pos&neg data
        vis_cfg = self.cfg.datasets.cc_sbu_align.vis_processor.train
        self.vis_processor = self._build_proc_from_cfg(vis_cfg)
        print('Load Augmented Caption Data...')
        self.data = json.load(open(self.data_path))
        
        self.sample_strategy = sample_strategy
        print(f"sampleing strategy: {self.sample_strategy}")
        if self.sample_strategy == "offline":
            for index in range(len(self.data)):
                self.data[index]["chosen"] = [random.choice(self.data[index]["chosen"])]
                self.data[index]["rejected"] = [random.choice(self.data[index]["rejected"])]
        
        print(f"Loaded {len(self.data)} description data")
        
        print("Data example:")
        chosen, rejected = self.data[0]["chosen"][0], self.data[0]["rejected"][0]
        print(f"Chosen: {chosen}")
        print(f"Rejected: {rejected}")
        
        # visual genome
        self.vg_image_data = json.load(open(os.path.join(self.vg_path, "image_data.json")))
        self.id2path = {
            _data["image_id"]:os.path.join(self.vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
            for _data in self.vg_image_data
        }
        
        if self.auxilary_dataset is not None:
            if "LOCAL_RANK" not in os.environ:
                self.aux_start, self.aux_end = 0, len(self.auxilary_dataset) - 1
            else:
                step = len(self.auxilary_dataset)//int(os.environ["WORLD_SIZE"]) + 1
                local_rank = int(os.environ["LOCAL_RANK"])
                self.aux_start = step * local_rank
                self.aux_end = min(step * (local_rank + 1), len(self.auxilary_dataset))
            self.aux_idx_generator = (idx for idx in range(self.aux_start, self.aux_end))
        
    def _build_proc_from_cfg(self, cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )
        
    def load_data(self, index):
        anno = self.data[index]
        image_id = anno["image_id"]
        image_path = self.id2path[int(image_id)]
        if self.sample_strategy == "online":
            chosen = random.choice(anno["chosen"])
            rejected = random.choice(anno["rejected"])
        elif self.sample_strategy == "offline":
            chosen = anno["chosen"][0]
            rejected = anno["rejected"][0]
        return {
            "image_id": image_id,
            "image_path": image_path,
            "image": self.vis_processor(Image.open(image_path).convert("RGB")),
            "chosen": chosen.replace("\n", ""),
            "rejected": rejected.replace("\n", ""),
            "prompt": "",   # For data  collator 
            "data_type": "desc",
        }
        
    def select(self, indices):
        return [
            self.__getitem__(idx) for idx in indices
        ]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        sample = self.load_data(index)
        if self.auxilary_dataset is not None:
            try:
                aux_idx = next(self.aux_idx_generator)
            except BaseException:
                self.aux_idx_generator = (idx for idx in range(self.aux_start, self.aux_end))
                aux_idx = next(self.aux_idx_generator)
            aux_sample = self.auxilary_dataset[aux_idx]
            return {
                k: [sample[k], aux_sample[k]] for k in sample.keys()
            }
        else:
            return {
                k: [sample[k]] for k in sample.keys()
            }
        
        
class CCSBUAlignDataset(Dataset):
    """
        PopeDataset:
        use GPT-4 reconstructed POPE-format dataset.
    """
    def __init__(self,
        ccsbualign_data_path: str,
        cfg: OmegaConf,
    ):
        super(CCSBUAlignDataset, self).__init__()
        self.data_path = ccsbualign_data_path
        self.annotation = json.load(open(os.path.join(self.data_path, "filter_cap.json")))
        self.cfg = cfg
        
        # pos&neg data
        vis_cfg = self.cfg.datasets.cc_sbu_align.vis_processor.train
        self.vis_processor = self._build_proc_from_cfg(vis_cfg)
        
        self.load_data()
        
    def _build_proc_from_cfg(self, cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )
        
    def load_data(self):
        self.data_list = []
        print('Load CCSBUAlign Data...')
        for anno in self.annotation["annotations"]:
            image_id = anno["image_id"]
            image_path = os.path.join(self.data_path, "image", f"{image_id}.jpg")
            self.data_list.append({
                "image_id": image_id,
                "image_path": image_path,
                "image": self.vis_processor(Image.open(image_path).convert("RGB")),
                "chosen": anno["caption"],
                "rejected": "",
                "data_type": "ccsbualign",
                "prompt": "",
            })
                
        print(f"Loaded {len(self.data_list)} CCSBUAlign data")
        print("Data Example:")
        print(self.data_list[0]["chosen"])
        
    def select(self, indices):
        return [
            self.__getitem__(idx) for idx in indices
        ]
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, index):
        return self.data_list[index]