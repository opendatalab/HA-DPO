from .vigc_train import LlavaBaseDataset
import json
import random


class COCO_Pseudo_Dataset(LlavaBaseDataset):
    VIG_INSTRUCTIONS = (
        "Based on the content of the given image, generate a question that requires common sense to answer and then briefly answer it.",
        "Explain the content of the image in a question and then provide a short answer using knowledge types such as commonsense and facts.",
        "Generate a query that requires reasoning on the information depicted in the image, utilizing a variety of knowledge types like commonsense, and then offer a concise answer.",
        "Develop a query to demonstrate the knowledge types such as commonsense and facts related to the given image and then provide a brief answer.",
        "Based on knowledge types such as commonsense and facts, come up with a query related to the given image and then briefly answer it.",
        "Come up with a question related to the content shown in the image that requires reasoning using a variety of knowledge types such as commonsense and then succinctly answer it.",
        "Brainstorm a question about the content of the given image that requires reasoning with a variety of knowledge types such as common sense and then state the answer briefly.",
        "Construct a query that requires logic based on the contents of the given image and involves a variety of knowledge types such as commonsense, and then deliver a brief response.",
        "Invent an inquiry derived from the pictured material that calls for the use of different knowledge types like commonsense and subsequently summarize the solution with brevity.",
    )

    def __init__(self, vis_processor, text_processor, vis_root, anno_path, task: str, image_ids_file, filter, topk,
                 threshold):
        self.topk = topk
        self.threshold = threshold
        super().__init__(vis_processor, text_processor, vis_root, anno_path, task)
        image_ids = json.load(open(image_ids_file))
        _filter_image_ids = {k: v for k, v in image_ids.items() if k in filter}
        filter_image_ids = []
        for ids in _filter_image_ids.values():
            filter_image_ids.extend(ids)
        self.filter_image_ids = set(filter_image_ids)
        filtered_samples = []

        for sample in self.samples:
            if sample["image_id"] not in filter_image_ids:
                filtered_samples.append(sample)
        try:
            if threshold > 0:
                self.samples = [_ for _ in filtered_samples if max(_["weight"]) > threshold]
            else:
                self.samples = filtered_samples
        except KeyError:
            self.samples = filtered_samples

    def get_qa_image(self, ann):
        image = self.vis_processor(self._read_image(ann))
        question = self.text_processor(ann["question"])
        answers = ann["answer"]
        if isinstance(answers, str):
            answer = self.text_processor(ann["answer"])
        else:
            weights = ann['weight'][:self.topk]
            answer = self.text_processor(random.choices(answers[:self.topk], weights=weights, k=1)[0])
        return {"image": image, "question": question, "answer": answer}
