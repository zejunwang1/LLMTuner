# coding=utf-8

import torch
import transformers
from dataclasses import dataclass, field
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer,
    HfArgumentParser,
    set_seed
)
from dataset import Dataset, DataCollator
from trainer import Trainer

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HuggingFace model name or path."})
    model_type: str = field(default=None, metadata={"help": "Base model type: llama or bloom."})

@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to the training data."})
    eval_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str = field(default=None)
    optim: str = field(default='adamw_torch', metadata={"help": 'The optimizer to be used.'})
    use_flash_attn: bool = field(default=False, metadata={"help": "Whether use flash attention for training."})
    use_xformers_attn: bool = field(
        default=False,
        metadata={"help": "Whether use xformers attention for training."}
    )
    task_finetune: bool = field(default=False, metadata={"help": "Enable task-specific fine-tuning."})
    task_prompt: str = field(default=None, metadata={"help": "Description of specific task."})

def make_supervised_data_module(data_args, training_args, tokenizer):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Dataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        task_finetune=training_args.task_finetune,
        task_prompt=training_args.task_prompt,
        max_length=data_args.max_length
    )

    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = Dataset(
            data_path=data_args.eval_path,
            tokenizer=tokenizer,
            task_finetune=training_args.task_finetune,
            task_prompt=training_args.task_prompt,
            max_length=data_args.max_length
        )
    
    data_collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()    
    # set seed
    set_seed(training_args.seed)
    
    # replace llama attention with flash attention
    assert not (training_args.use_flash_attn and training_args.use_xformers_attn)
    if model_args.model_type == "llama" and training_args.use_flash_attn:
        from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # replace llama attention with xformers attention
    if model_args.model_type == "llama" and training_args.use_xformers_attn:
        from llama_xformers_attn_monkey_patch import replace_llama_attn_with_xformers_attn
        replace_llama_attn_with_xformers_attn()

    # load model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )
    # Qwen natively supports flash attention
    if hasattr(config, "use_flash_attn"):
        setattr(config, "use_flash_attn", training_args.use_flash_attn) 
    
    print(f'loading base model {model_args.model_name_or_path}...')
    torch_dtype = (torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32))
    setattr(config, "torch_dtype", torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        trust_remote_code=True
    )
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.bos_token = '<|im_start|>'
        tokenizer.eos_token = '<|im_end|>'
        tokenizer.pad_token_id = tokenizer.eod_id
    else:
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        if tokenizer.pad_token_id is None:
            assert tokenizer.unk_token_id is not None
            tokenizer.pad_token_id = tokenizer.unk_token_id

    # prepare dataset
    data_module = make_supervised_data_module(data_args, training_args, tokenizer)
    
    # train
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()

