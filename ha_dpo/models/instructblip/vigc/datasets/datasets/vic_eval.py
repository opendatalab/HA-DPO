import random
import torch
from .base_dataset import BaseDataset


class A_OKVQAEvalDataset(BaseDataset):
    PROMPTS = (
        "Question: {q} Short answer:",
    )

    def __init__(self, vis_processor, text_processor, vis_root, anno_file):
        super().__init__(vis_processor, text_processor, vis_root, anno_file)

    def __getitem__(self, index):
        ann = self.samples[index]

        image = self.vis_processor(self._read_image(ann, image_key="image_path"))
        question = self.text_processor(ann["question"])

        answer_weight = {}
        for answer in ann["direct_answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["direct_answers"])
            else:
                answer_weight[answer] = 1 / len(ann["direct_answers"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())
        prompt = random.choice(self.PROMPTS)
        question = prompt.format(q=question)

        choices = ann["choices"]
        correct_choice_idx = ann["correct_choice_idx"]

        raw_sample = {"img_path": ann["image_path"], "question": ann["question"], "gt_answer": answers,
                      "gt_answer_weight": weights, "image_id": ann["image_id"], "choices": choices,
                      "correct_choice_idx": correct_choice_idx, "gt_answer_weight_rank": [1.0],
                      "gt_answer_rank": [choices[correct_choice_idx]]}

        if "question_id" in ann:
            raw_sample["question_id"] = ann["question_id"]
        input_sample = {
            "image": image,
            "prompt": question
        }
        return input_sample, raw_sample

    def collater(self, samples):
        image_list, prompt_list, raw_sample_list, candidates = [], [], [], []
        for input_sample, raw_sample in samples:
            raw_sample_list.append(raw_sample)
            image_list.append(input_sample["image"])
            prompt_list.append(input_sample["prompt"])
            candidates.append(raw_sample.get("choices", []))

        return {
            "image": torch.stack(image_list, dim=0),
            "prompt": prompt_list,
            "candidates": candidates,
            "raw_samples": raw_sample_list
        }


class A_OKVQATestDataset(BaseDataset):
    PROMPTS = (
        "Question: {q} Short answer:",
    )

    def __init__(self, vis_processor, text_processor, vis_root, anno_file):
        super().__init__(vis_processor, text_processor, vis_root, anno_file)

    def __getitem__(self, index):
        ann = self.samples[index]

        image = self.vis_processor(self._read_image(ann))
        question = self.text_processor(ann["question"])

        prompt = random.choice(self.PROMPTS)
        question = prompt.format(q=question)

        choices = ann["choices"]

        raw_sample = {
            "image": ann["image"], "question": ann["question"], "choices": choices, "question_id": ann["question_id"]
        }

        input_sample = {
            "image": image,
            "prompt": question
        }
        return input_sample, raw_sample

    def collater(self, samples):
        image_list, prompt_list, raw_sample_list, candidates = [], [], [], []
        for input_sample, raw_sample in samples:
            raw_sample_list.append(raw_sample)
            image_list.append(input_sample["image"])
            prompt_list.append(input_sample["prompt"])
            candidates.append(raw_sample.get("choices", []))

        return {
            "image": torch.stack(image_list, dim=0),
            "prompt": prompt_list,
            "candidates": candidates,
            "raw_samples": raw_sample_list
        }


class VQAv2EvalDataset(A_OKVQAEvalDataset):
    PROMPTS = (
        "Question: {q} Short answer:",
    )

    def __init__(self, vis_processor, text_processor, vis_root, anno_file, sample_ratio=1):
        super().__init__(vis_processor, text_processor, vis_root, anno_file)
        self.samples = self.samples[::sample_ratio]

    def __getitem__(self, index):
        ann = self.samples[index]

        image = self.vis_processor(self._read_image(ann))
        question = self.text_processor(ann["question"])

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())
        prompt = random.choice(self.PROMPTS)
        question = prompt.format(q=question)

        image_path = ann["image"]
        image_id = int(image_path.split("_")[-1][:-4])

        raw_sample = {"img_path": ann["image"], "question": ann["question"], "gt_answer": answers,
                      "gt_answer_weight": weights, "image_id": image_id}
        if "question_id" in ann:
            raw_sample["question_id"] = ann["question_id"]
        input_sample = {
            "image": image,
            "prompt": question
        }
        return input_sample, raw_sample
