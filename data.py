import json
import re
from datasets import load_dataset,Dataset

"""
3 stages of training that come from 3 format of dataset
1. 1 Image+Instruction+Question -> Evidences (ie_process)
2. 1 Image(frozen)+Question + Evidences as instruction -> Answers (qea_process)
3. Images + Question -> Evidences + Answers (preprocess_fn)
"""

def extract_regions(text):
    return re.findall(r"<region>.*?</region>",text,flags=re.DOTALL)

def qea_process(example):
    """
    this is format for vlm to lean to reason and answer from image,evidence, by using regions data 1 image contain multiple evidences all difference depends on question
    :param example:
    :return:
    """
    question = example['question']
    answer = example['answer']
    reason = example['reason']
    regions = example['regions']
    regions = json.loads(regions)
    evidence_str = example['evidence']
    # map image and region together
    info = []



    for data in regions:
        conversations = []
        user = {
            "role": "user",
            "content": [],
        }
        assistant = {
            "role": "assistant",
            "content": [],
        }
        region_idx = data['region_idx']
        evidence = extract_regions(evidence_str)
        if evidence and len(evidence) > region_idx:
            evidence_per_region = evidence[region_idx]
        else:
            evidence_per_region = ""

        # user question. it is the same question for multiple image in sample the answer will be by image evidence
        user['content'] = [
            {"type": "text","text":evidence_per_region+question}
        ]
        # evidence as answer
        assistant['content'] = [{"type": "text", "text": reason+answer}]

        conversations.append(user)
        conversations.append(assistant)

        info.append({
            "message":conversations,
        })

    return info

def ie_process(example):
    """
    this is format for vlm to lean to extract evidences from image, by using regions data 1 image contain multiple evidences all difference depends on question
    :param example:
    :return:
    """
    images = example['images']
    instruction = example['instruction']
    question = example['question']
    masks = example['masks']
    regions = example['regions']
    regions = json.loads(regions)
    evidence_str = example['evidence']
    # map image and region together
    info = []



    for data in regions:
        conversations = []
        user = {
            "role": "user",
            "content": [],
        }
        assistant = {
            "role": "assistant",
            "content": [],
        }
        # map index individual
        image_idx = data['image_idx']
        mask_idx = data['mask_idx']
        assert(image_idx == mask_idx) # check if it same indexes from original dataset
        region_idx = data['region_idx']
        evidence = extract_regions(evidence_str)
        if evidence and len(evidence) > region_idx:
            evidence_per_region = evidence[region_idx]
        else:
            evidence_per_region = ""

        # user question. it is the same question for multiple image in sample the answer will be by image evidence

        user['content'] = [
            {"type": "image"},{"type": "text","text": instruction+question}
        ]
        # evidence as answer
        assistant['content'] = [{"type": "text", "text": evidence_per_region}]

        conversations.append(user)
        conversations.append(assistant)

        info.append({
            "i":images[image_idx],
            "m":masks[mask_idx],
            "message":conversations,
        })
    return info


def preprocess_fn(example):
    images = example['images']
    masks = example['masks']

    instruction = example['instruction']
    question = example['question']
    answer = example['answer']
    # reason = example['reason']
    evidence = example['evidence']

    user = {
        "role":"user",
        "content":[],
    }
    assistant = {
        "role": "assistant",
        "content": [],
    }

    conversations = []

    il,ml = [],[]
    if images and masks:
        if isinstance(images,list) and isinstance(masks,list):
            assert len(images) == len(masks)

            for i,m in zip(images,masks):
                il.append(i)
                ml.append(m)

    if instruction and question:
        content = []
        if isinstance(images,list) and isinstance(masks,list):
            for _ in range(len(images)):
                content.append({"type":"image"})
        content.append({
            "type":"text",
            "text":instruction+question,
        })
        user['content'] = content

    if evidence and answer:
        content = []
        concat_str = evidence + answer
        content.append({"type":"text","text":concat_str})
        assistant['content'] = content

    conversations.append(user)
    conversations.append(assistant)

    text = {"message":conversations}
    images = {"i":il,"m":ml}
    return text | images

class VisionCollator:
    def __init__(self,processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, example):
        image = [ensure_image_list(image['i']) for image in example] # each key contains image so it is list of list
        messages = [clean_messages(text['message']) for text in example]
        texts = [self.tokenizer.apply_chat_template(
            message,tokenize=False,add_generation_prompt=False
        ) for message in messages]
        prompt_texts = [self.tokenizer.apply_chat_template(
            message[:1],tokenize=False,add_generation_prompt=True
        ) for message in messages]

        # mask =[mask['m'] for mask in example]

        batch = self.processor(text=texts, images=image, padding=True
                               ,return_tensors="pt")

        batch['labels'] = build_labels(
            tokenizer=self.tokenizer,
            input_ids=batch['input_ids'],
            prompt_texts=prompt_texts,
        )
        return batch

class LangCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, example):

        messages = [clean_messages(text['message']) for text in example]
        texts = [self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        ) for message in messages]
        prompt_texts = [self.tokenizer.apply_chat_template(
            message[:1], tokenize=False, add_generation_prompt=True
        ) for message in messages]


        batch = self.processor(text=texts, padding=True
                               , return_tensors="pt")

        batch['labels'] = build_labels(
            tokenizer=self.tokenizer,
            input_ids=batch['input_ids'],
            prompt_texts=prompt_texts,
        )
        return batch


def ensure_image_list(value):
    if isinstance(value, list):
        return value
    return [value]

def clean_messages(messages):
    cleaned = []
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] == "image":
                content.append({"type": "image"})
            elif item["type"] == "text":
                content.append({"type": "text", "text": item.get("text") or ""})
        cleaned.append({"role": message["role"], "content": content})
    return cleaned

def build_labels(tokenizer, input_ids, prompt_texts):
    labels = input_ids.clone()
    prompt_inputs = tokenizer(
        prompt_texts,
        padding=False,
        add_special_tokens=False,
    )

    for row_idx, prompt_ids in enumerate(prompt_inputs["input_ids"]):
        labels[row_idx, :len(prompt_ids)] = -100

    labels[labels == tokenizer.pad_token_id] = -100
    return labels

class TemplateDataset:
    def __init__(self,dataset,test_ratio=0.2,map_fn=None):
        self.dataset_repo = dataset
        self.ds = load_dataset(self.dataset_repo)
        self.usable = self.ds['train']
        if map_fn is not None:
            rows = []
            for example in self.usable:
                processed = map_fn(example)
                if isinstance(processed, list):
                    rows.extend(processed)
                else:
                    rows.append(processed)
            self.temped_dataset = Dataset.from_list(rows)
        else:
            self.temped_dataset = self.usable
        hold = self.temped_dataset.train_test_split(test_size=test_ratio)
        test = hold['test'].train_test_split(0.5)
        self.train_dataset = hold['train']
        self.test_dataset = test['train']
        self.eval_dataset = test['test']

        self.column_feature = self.temped_dataset.features
