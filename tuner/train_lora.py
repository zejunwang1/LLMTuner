# coding=utf-8

import os
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
from peft import (
    TaskType,
    PeftModel,
    LoraConfig,
    get_peft_model
)
from dataset import Dataset, DataCollator
from trainer import LoraTrainer

@dataclass
class ModelArguments:
    model_type: str = field(metadata={"help": "Base model type: llama, qwen or baichuan."})
    model_name_or_path: str = field(metadata={"help": "HuggingFace model name or path."})
    target_modules: str = field(metadata={"help": "Lora target modules."})

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
    bits: int = field(default=None, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help":"Lora dropout."})
    additional_trainable_params: str = field(
        default=None,
        metadata={"help": "Additional trainable parameters except LoRA weights."}
    )

def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, use_flash_attn=False):
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if not use_flash_attn and not is_gptq_quantized:
        # cast all non INT8 parameters to fp32
        for param in model.parameters():
            if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                param.data = param.data.to(torch.float32)
    
    if (loaded_in_kbit or is_gptq_quantized) and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model

def get_accelerate_model(model_args, training_args):
    # DDP
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": training_args.local_rank}

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
        load_in_8bit=training_args.bits == 8,
        load_in_4bit=training_args.bits == 4,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    # prepare model for kbit training
    if training_args.bits in [4, 8]:
        prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # enable additional trainable params
    if training_args.additional_trainable_params:
        [p.requires_grad_() for n, p in model.named_parameters() 
            if any([k in n for k in training_args.additional_trainable_params.split(",")])]
    
    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        model.enable_input_require_grads()
    
    # Get our peft model and print the number of trainable parameters
    checkpoint_dir = training_args.resume_from_checkpoint
    if checkpoint_dir is not None:
        print(f"Resuming from {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
    else:
        print(f"adding LoRA modules...")
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=model_args.target_modules.split(','),
            lora_dropout=training_args.lora_dropout,
            bias="none",
            inference_mode=False,
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model

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
    model = get_accelerate_model(model_args, training_args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        trust_remote_code=True
    )
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.bos_token = '<|im_start|>'
        tokenizer.eos_token = '<|endoftext|>'
        tokenizer.pad_token = '<|im_end|>'
    else:
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        if tokenizer.pad_token_id is None:
            assert tokenizer.unk_token_id is not None
            tokenizer.pad_token_id = tokenizer.unk_token_id

    # prepare dataset
    data_module = make_supervised_data_module(data_args, training_args, tokenizer)

    # train
    trainer = LoraTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()

