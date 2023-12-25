# coding=utf-8

import transformers
from dataclasses import dataclass, field
from dataset import T5Dataset, DataCollator
from transformers import T5Tokenizer, T5ForConditionalGeneration, HfArgumentParser, set_seed
from trainer import Trainer

@dataclass
class DataArguments:
    model_name_or_path: str = field(metadata={"help": "HuggingFace model name or path."})
    data_path: str = field(metadata={"help": "Path to the training data."})
    eval_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    max_source_length: int = field(default=512, metadata={"help": "The maximum length of tokenized source."})
    max_target_length: int = field(default=512, metadata={"help": "The maximum length of tokenized target."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str = field(default=None)
    optim: str = field(default='adamw_torch', metadata={"help": 'The optimizer to be used.'})
    task_prompt: str = field(default=None, metadata={"help": "Description of specific task."})

def make_supervised_data_module(data_args, training_args, tokenizer):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = T5Dataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        task_prompt=training_args.task_prompt,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )

    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = Dataset(
            data_path=data_args.eval_path,
            tokenizer=tokenizer,
            task_prompt=training_args.task_prompt,
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length
        )

    data_collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

def train():
    parser = HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    # set seed
    set_seed(training_args.seed)

    # load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        data_args.model_name_or_path, cache_dir=training_args.cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(
        data_args.model_name_or_path, cache_dir=training_args.cache_dir)

    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    # prepare dataset
    data_module = make_supervised_data_module(data_args, training_args, tokenizer)

    # train
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()

