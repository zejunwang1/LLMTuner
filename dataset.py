# coding=utf-8

import json
import torch
from dataclasses import dataclass
from tqdm import tqdm

def load_dataset(data_path, tokenizer, max_length = 1024):
    output = []
    with open(data_path, mode="r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="processing"):
            data = json.loads(line.rstrip())
            conversation = data["conversation"]
            input_ids = []
            labels = []
            for chat in conversation:
                q_ids = tokenizer(chat["human"]).input_ids
                a_ids = tokenizer(chat["assistant"]).input_ids
                chat_ids = q_ids + [tokenizer.eos_token_id] + a_ids + [tokenizer.eos_token_id]
                q_len = len(q_ids) + 1
                chat_labels = [-100] * q_len + a_ids + [tokenizer.eos_token_id]
                input_ids += chat_ids
                labels += chat_labels

            # truncation
            if len(input_ids) > max_length:
                input_ids = input_ids[: max_length]
                labels = labels[: max_length]
            output.append(dict(input_ids=input_ids, labels=labels))
    return output

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_length = 1024):
        super(Dataset, self).__init__()
        self.data = load_dataset(data_path, tokenizer, max_length)
    
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
            labels, batch_first=True, padding_value=-100
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id)
        )


