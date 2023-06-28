# coding=utf-8

import bitsandbytes as bnb
import transformers
import torch
from dataclasses import dataclass, field
from dataset import Dataset, DataCollator
from trainer import LoraTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
from peft.tuners.lora import LoraLayer

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
    optim: str = field(default="paged_adamw_32bit")
    lora_r: int = field(default=64,
        metadata={"help": "Lora attention dimension."})
    lora_alpha: int = field(default=16,
        metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: float = field(default=0.05,
        metadata={"help": "The dropout probability for Lora layers."})
    bits: int = field(default=4, 
        metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})


def find_all_linear_names(model, bits):
    cls = bnb.nn.Linear4bit if bits == 4 else \
        (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_accelerate_model(model_args, training_args):
    # DDP
    device_map = "auto"
    if training_args.local_rank != -1:
        device_map = {"": training_args.local_rank}
    
    compute_dtype = (torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32))
    torch_dtype = (torch.float32 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32))
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=training_args.bits == 4,
        load_in_8bit=training_args.bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_use_double_quant=training_args.double_quant,
        bnb_4bit_quant_type=training_args.quant_type,
        bnb_4bit_compute_dtype=compute_dtype
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        load_in_4bit=training_args.bits == 4,
        load_in_8bit=training_args.bits == 8,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    model.config.torch_dtype = torch_dtype

    # cast all non INT8 parameters to fp32
    model = prepare_model_for_kbit_training(model,
        use_gradient_checkpointing=training_args.gradient_checkpointing)
    
    # Get our peft model and print the number of trainable parameters
    checkpoint_dir = training_args.resume_from_checkpoint
    if checkpoint_dir is not None:
        print(f"Resuming from {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
    else:
        modules = find_all_linear_names(model, training_args.bits)
        config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=modules,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            inference_mode=False,
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, config)
    
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    
    model.print_trainable_parameters()
    return model

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
    model = get_accelerate_model(model_args, training_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    data_module = make_supervised_data_module(data_args=data_args, tokenizer=tokenizer)
    trainer = LoraTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()

