# coding=utf-8

import torch
import transformers
from dataset import Dataset, DataCollator
from dataclasses import dataclass, field
from trainer import Trainer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    HfArgumentParser,
    set_seed
)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HuggingFace model name or path."})

@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to the training data."})
    eval_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    max_length: int = field(default=1024, metadata={"help": "Maximum length of input."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str = field(default=None)
    optim: str = field(default="adamw_torch")

def make_supervised_data_module(data_args, tokenizer):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Dataset(data_args.data_path, tokenizer, data_args.max_length)
    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = Dataset(data_args.eval_path, tokenizer, data_args.max_length)
    
    data_collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    return dict(train_dataset=train_dataset, 
                eval_dataset=eval_dataset, 
                data_collator=data_collator)
        
def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(training_args.seed)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(data_args=data_args, tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
if __name__ == "__main__":
    train()

