# coding=utf-8

import json
import torch
from dataclasses import dataclass
from tqdm import tqdm

IGNORE_INDEX = -100

def load_t5_dataset(data_path, tokenizer, task_prompt=None, max_source_length=512, max_target_length=512):
    output = []
    if task_prompt is not None:
        task_prompt += "{}"
    with open(data_path, mode="r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="processing"):
            data = json.loads(line.strip())
            source = task_prompt.format(data["source"]) if task_prompt is not None else data["source"]
            input_ids = tokenizer.encode(source, max_length=max_source_length, truncation=True)
            labels = tokenizer.encode(data["target"], max_length=max_target_length, truncation=True)
            output.append(dict(input_ids=input_ids, labels=labels))
    return output

def load_task_dataset(data_path, tokenizer, task_prompt=None, max_length=1024):
    output = []
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    if tokenizer.__class__.__name__ == "ChatGLMTokenizer":
        bos_token_id = tokenizer.get_command("<bos>")
    if task_prompt is not None:
        task_prompt += "{}"
    with open(data_path, mode="r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="processing"):
            data = json.loads(line.strip())
            # tokenization
            # f"<s>{source}</s><s>{target}</s>"
            source = task_prompt.format(data["source"]) if task_prompt is not None else data["source"]
            source_ids = tokenizer.encode(source, add_special_tokens=False)
            target_ids = tokenizer.encode(data["target"], add_special_tokens=False)
            input_ids = [bos_token_id] + source_ids + [eos_token_id] + \
                        [bos_token_id] + target_ids + [eos_token_id]
            ignore_len = len(source_ids) + 3
            labels = [IGNORE_INDEX] * ignore_len + target_ids + [eos_token_id]
            # truncation
            if len(input_ids) > max_length:
                input_ids = input_ids[: max_length]
                labels = labels[: max_length]
            output.append(dict(input_ids=input_ids, labels=labels))
    return output

def load_instruction_dataset(data_path, tokenizer, max_length=1024):
    output = []
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    if tokenizer.__class__.__name__ == "ChatGLMTokenizer":
        bos_token_id = tokenizer.get_command("<bos>")
    with open(data_path, mode="r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="processing"):
            data = json.loads(line.strip())
            conversation = data["conversation"]
            input_ids = []
            labels = []
            # multiple rounds of dialogue
            # f"<s>{human}</s><s>{assistant}</s><s>{human}</s><s>{assistant}</s>"
            for chat in conversation:
                q_ids = tokenizer.encode(chat["human"], add_special_tokens=False)
                a_ids = tokenizer.encode(chat["assistant"], add_special_tokens=False)
                chat_ids = [bos_token_id] + q_ids + [eos_token_id] + \
                           [bos_token_id] + a_ids + [eos_token_id]
                ignore_len = len(q_ids) + 3
                chat_labels = [IGNORE_INDEX] * ignore_len + a_ids + [eos_token_id]
                input_ids += chat_ids
                labels += chat_labels
            # truncation
            if len(input_ids) > max_length:
                input_ids = input_ids[: max_length]
                labels = labels[: max_length]
            output.append(dict(input_ids=input_ids, labels=labels))
    return output

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, tokenizer, task_finetune=False, task_prompt=None, max_length=1024
    ):
        super(Dataset, self).__init__()
        if task_finetune:
            self.data = load_task_dataset(data_path, tokenizer, task_prompt, max_length)
        else:
            self.data = load_instruction_dataset(data_path, tokenizer, max_length)

    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Basic function of `Dataset` to get sample from dataset with a given index.
        """
        return self.data[index]

class T5Dataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, tokenizer, task_prompt=None, max_source_length=512, max_target_length=512
    ):
        super(Dataset, self).__init__()
        self.data = load_t5_dataset(data_path, tokenizer, task_prompt, max_source_length, max_target_length)

    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Basic function of `Dataset` to get sample from dataset with a given index.
        """
        return self.data[index]

@dataclass
class DataCollator:
    """
    Collate examples for supervised fine-tuning.
    """
    pad_token_id: int = 0

    def __call__(self, instances):
        input_ids, labels = tuple([torch.tensor(instance[key]) for instance in instances]
                            for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id).long()
        )

