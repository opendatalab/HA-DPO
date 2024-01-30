import os
import tqdm
import json
import random
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from vigc.common.config import Config
from vigc.common.registry import registry

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
    ):
        super(AugmentedCaptionDataset, self).__init__()
        
        random.seed(seed)
        self.data_path = data_path
        self.vg_path = vg_path
        self.cfg = cfg
        
        # pos&neg data
        vis_cfg = self.cfg.datasets.instruct_blip_given_q_coco2017_vig_test.vis_processor.train
        self.vis_processor = self._build_proc_from_cfg(vis_cfg)
        print('Load Description Data...')
        self.data = json.load(open(self.data_path))
        
        self.sample_strategy = sample_strategy
        print(f"sampleing strategy: {self.sample_strategy}")
        assert self.sample_strategy in ["online", "offline"]
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
        chosen = random.choice(anno["chosen"])
        rejected = random.choice(anno["rejected"])
        return {
            "image_id": image_id,
            "image_path": image_path,
            "image": self.vis_processor(Image.open(image_path).convert("RGB")),
            "chosen": chosen.replace("\n", ""),
            "rejected": rejected.replace("\n", ""),
            "prompt": random.choice([
                "Describe this image in detail.",
                "Take a look at this image and describe what you notice.",
                "Please provide a detailed description of the picture.",
                "Could you describe the contents of this image for me?",
            ]),
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
        return sample
    
    
class PopeDataset(Dataset):
    """
        PopeDataset:
        use GPT-4 reconstructed POPE-format dataset.
    """
    def __init__(self, 
        data_path: str,
        vg_path: str,
        cfg: OmegaConf,
    ):
        super(PopeDataset, self).__init__()
        self.data_path = data_path
        self.vg_path = vg_path
        self.cfg = cfg
        
        # pos&neg data
        vis_cfg = self.cfg.datasets.instruct_blip_given_q_coco2017_vig_test.vis_processor.train
        self.vis_processor = self._build_proc_from_cfg(vis_cfg)
        self.data = json.load(open(self.data_path))
        
        # visual genome
        self.vg_image_data = json.load(open(os.path.join(self.vg_path, "image_data.json")))
        self.id2path = {
            _data["image_id"]:os.path.join(self.vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
            for _data in self.vg_image_data
        }
        self.load_data()
        
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
            if anno["correct"]:
                continue
            image_id = anno["image_id"]
            image_path = self.id2path[int(image_id)]
            self.data_list.append({
                "image_id": image_id,
                "image_path": image_path,
                "image": self.vis_processor(Image.open(image_path).convert("RGB")),
                "chosen": anno["chosen"],
                "rejected": anno["rejected"],
                "data_type": "pope",
                "prompt": anno["question"],
            })
                
        print(f"Loaded {len(self.data_list)} pope data")
        
        print("Data Example:")
        print("Chosen: ", self.data_list[0]["chosen"])
        print("Rejected: ", self.data_list[0]["rejected"])
        
    def select(self, indices):
        return [
            self.__getitem__(idx) for idx in indices
        ]
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, index):
        sample = self.data_list[index]
        return sample