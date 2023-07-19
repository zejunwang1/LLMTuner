# coding=utf-8

import os
import transformers
from dataclasses import dataclass, field
from dataset import Dataset, DataCollator
from trainer import LoraTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed
)

from peft import (
    TaskType,
    PeftModel,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

@dataclass
class ModelArguments:
    model_type: str = field(metadata={"help": "Base model type: bloom or llama."})
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
    lora_r: int = field(default=64, 
        metadata={"help": "Lora attention dimension."})
    lora_alpha: int = field(default=16, 
        metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: float = field(default=0.05, 
        metadata={"help": "The dropout probability for Lora layers."})
    int8_training: bool = field(default=False,
        metadata={"help": "If True, enable INT8 training."})
    int4_training: bool = field(default=False,
        metadata={"help": "If True, enable INT4 training."})
    
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
    
    # DDP
    device_map = "auto"
    if training_args.local_rank != -1:
        device_map = {"": training_args.local_rank}
    
    # Load model and tokenizer
    assert not (training_args.int8_training and training_args.int4_training)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        load_in_8bit=training_args.int8_training,
        load_in_4bit=training_args.int4_training,
        device_map=device_map,
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
    
    # Define LoRA Config
    modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_args.model_type]
    config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=modules,
        lora_dropout=training_args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )
    
    # cast all non INT8 parameters to fp32
    model = prepare_model_for_kbit_training(model, 
        use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
    # Get our peft model and print the number of trainable parameters
    checkpoint_dir = training_args.resume_from_checkpoint
    if checkpoint_dir is not None:
        print(f"Resuming from {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
    else:
        model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(data_args=data_args, tokenizer=tokenizer)
    trainer = LoraTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()

