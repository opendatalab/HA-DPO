from .base_dataset import BaseDataset
import random


class LlavaBaseDataset(BaseDataset):
    VIG_INSTRUCTIONS = None
    VIG_PROMPTS = (
        " Question: {q} Answer: {a}",
    )

    VQA_PROMPTS = (
        "{q}",
        "Question: {q}",
        "{q} A short answer to the question is",
        "Q: {q} A:",
        "Question: {q} Short answer:",
        "Given the image, answer the following question with no more than three words. {q}",
        "Based on the image, respond to this question with a short answer: {q}. Answer:",
        "Use the provided image to answer the question: {q} Provide your answer as short as possible:",
        'What is the answer to the following question? "{q}"',
        'The question "{q}" can be answered using the image. A short answer is'
    )

    VQG_PROMPTS = (
        "Given the image, generate a question whose answer is: {a}. Question:",
        "Based on the image, provide a question with the answer: {a}. Question:",
        'Given the visual representation, create a question for which the answer is "{a}".',
        'From the image provided, craft a question that leads to the reply: {a}. Question:',
        'Considering the picture, come up with a question where the answer is: {a}.',
        'Taking the image into account, generate an question that has the answer: {a}. Question:'
    )

    def __init__(self, vis_processor, text_processor, vis_root, anno_path, task: str):
        assert task in ("vic", "vqg", "vig")
        self.task = task
        super(LlavaBaseDataset, self).__init__(vis_processor, text_processor, vis_root, anno_path)

    def get_vqa_sample(self, question, answer):
        prompt = random.choice(self.VQA_PROMPTS)
        text_input = prompt.format(q=question)
        text_output = answer
        return text_input, text_output

    def get_vig_sample(self, question, answer):
        instruction = random.choice(self.VIG_INSTRUCTIONS)
        prompt = random.choice(self.VIG_PROMPTS).format(q=question, a=answer)
        text_input, text_output = instruction, prompt
        return text_input, text_output

    def get_vqg_sample(self, question, answer):
        prompt = random.choice(self.VQG_PROMPTS)
        text_input = prompt.format(a=answer)
        text_output = question
        return text_input, text_output

    def get_qa_image(self, ann):
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
        answer = self.text_processor(random.choices(answers, weights=weights, k=1)[0])
        return {"image": image, "question": question, "answer": answer}

    def __getitem__(self, index):
        ann = self.samples[index]

        process_ann = self.get_qa_image(ann)
        image, question, answer = process_ann["image"], process_ann["question"], process_ann["answer"]

        if self.task == "vic":
            text_input, text_output = self.get_vqa_sample(question, answer)
        elif self.task == "vig":
            text_input, text_output = self.get_vig_sample(question, answer)
        else:  # vqg
            text_input, text_output = self.get_vqg_sample(question, answer)

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
        }


class LlavaCompDataset(LlavaBaseDataset):
    VQA_PROMPTS = (
        "{q}",
    )

    VIG_INSTRUCTIONS = (
        "Based on the given image, generate an in-depth reasoning question and then answer it.",
        "Given the image, generate an in-depth reasoning question and answer.",
        "Taking the image into account, generate an reasoning question along with the answer.",
        "Can you come up with a reasoning question based on the image and then provide the answer?",
        "After looking at the image, devise a reasoning question and provide the answer to it.",
        "Contemplate the image and create a reasoning question with the answer provided.",
        "Analyze the image and provide a reasoning question as well as the answer.",
        "Compose a reasoning question using the image with its answer.",
        "Evaluate the image and create a comprehensive reasoning question and its answer.",
        "Analyze the image and craft an effective reasoning question and its response.",
    )


class LlavaConvDataset(LlavaBaseDataset):
    VQA_PROMPTS = (
        "{q}",
    )
    VIG_INSTRUCTIONS = (
        "Generate a question based on the content of the given image and then answer it.",
        "Given the image, generate a question along with the answer.",
        "From the image provided, craft a question and answer it.",
        "Come up with a question related to the content of the image and provide the answer.",
        "Brainstorm a query associated to the image and provide the response.",
        "Construct a question based on the information presented in the image and answer it.",
        "Ask yourself a question about the content of the image and respond to it.",
        "Establish a query related to the content of the image and give the answer.",
        "Ask a question derived from the image and then answer it.",
        "Create a question about the image and answer it."
    )


class LlavaDescDataset(LlavaBaseDataset):
    VQA_PROMPTS = (
        "{q}",
    )
    VIG_INSTRUCTIONS = (
        "Generate a question to describe the image content in detail and then answer it.",
        "Considering the picture, come up with a question to describe the image content in detail along with the answer.",
        "Describe the image content with a question and give the response.",
        "Come up with a creative question to express the image content and then provide the answer.",
        "Draft a query to address the image content and give the reply.",
        "Create a question to reveal the image content and give the resolution.",
        "Given the photo, state a question that reveals the details of the image and then answer it.",
        "Ask a question about what is depicted in the image and then answer it.",
        "Make up a query to explain the photo in more detail and answer it.",
        "Compose a question describing the subject of the image, followed by the answer.",
    )


class A_OKVQADataset(LlavaBaseDataset):
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

    def get_qa_image(self, ann):
        image = self.vis_processor(self._read_image(ann, "image_path"))
        question = self.text_processor(ann["question"])
        all_answers = ann["direct_answers"]

        answer_weight = {}
        for answer in all_answers:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(all_answers)
            else:
                answer_weight[answer] = 1 / len(all_answers)

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())
        answer = self.text_processor(random.choices(answers, weights=weights, k=1)[0])
        return {"image": image, "question": question, "answer": answer}


class OKVQADataset(LlavaBaseDataset):
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


class VQAv2Dataset(LlavaBaseDataset):
    VIG_INSTRUCTIONS = (
        "Generate a question based on the content of the given image and then briefly answer it.",
        "Given the image, generate a question along with the short answer.",
        "From the image provided, craft a question and briefly answer it.",
        "Come up with a question related to the content of the image and provide the brief answer.",
        "Brainstorm a query associated to the image and provide the brief response.",
        "Construct a question based on the information presented in the image and briefly answer it.",
        "Ask yourself a question about the content of the image and briefly respond to it.",
        "Establish a query related to the content of the image and give the short answer.",
        "Ask a question derived from the image and then briefly answer it.",
        "Create a question about the image and briefly answer it."
    )
