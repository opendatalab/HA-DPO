import webdataset as wds
from .base_dataset import BaseDataset
from torch.utils.data import ChainDataset, IterableDataset
import random


class CaptionDataset(BaseDataset):
    PROMPTS = (
        "A short image caption:",
        "A short image description:",
        "A photo of",
        "An image that shows",
        "Write a short description for the image.",
        "Write a description for the photo.",
        "Provide a description of what is presented in the photo.",
        "Briefly describe the content of the image.",
        "Can you briefly explain what you see in the image?",
        "Could you use a few words to describe what you perceive in the photo?",
        "Please provide a short depiction of the picture.",
        "Using language, provide a short account of the image.",
        "Use a few words to illustrate what is happening in the picture.",
    )

    def __init__(self, vis_processor, text_processor, vis_root, anno_path):
        super(CaptionDataset, self).__init__(vis_processor, text_processor, vis_root, anno_path)

    def __getitem__(self, index):
        ann = self.samples[index]
        image = self.vis_processor(self._read_image(ann))

        caption = ann["caption"]
        if isinstance(caption, list):
            cap_weight = {}
            for cap in caption:
                if cap in cap_weight.keys():
                    cap_weight[cap] += 1 / len(caption)
                else:
                    cap_weight[cap] = 1 / len(caption)
            caption = self.text_processor(random.choices(caption, cap_weight, k=1)[0])
        elif isinstance(caption, str):
            caption = self.text_processor(caption)

        prompt = random.choice(self.PROMPTS)

        return {
            "image": image,
            "text_input": prompt,
            "text_output": caption
        }


class WdsCCSBUDataset:
    def __init__(self, vis_processor, text_processor, location):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": random.choice(CaptionDataset.PROMPTS),
            "text_output": self.text_processor(sample[1]["caption"]),
        }


class CCSBUDataset(ChainDataset):
    def __init__(self, vis_processor, text_processor, location, length=0, min_length=1):
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.min_length = min_length

        datasets = [wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )]

        super().__init__(datasets)
        self._length = length

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "caption": self.text_processor(sample[1]["caption"]),
        }

    def process_sample(self, sample):
        return {
            "image": sample["image"],
            "text_input": random.choice(CaptionDataset.PROMPTS),
            "text_output": sample["caption"]
        }

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                caption = x["caption"]
                caption_words = caption.split(" ")
                if len(caption_words) >= self.min_length:
                    yield self.process_sample(x)

    def __len__(self):
        return self._length
