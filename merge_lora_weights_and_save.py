# coding=utf-8

import argparse
import torch
import transformers
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--peft_model", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    return args

def merge_and_save(args):
    device_map = {"": 0} if args.device == "cuda" else {"": "cpu"}
    # load base model and tokenizer
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.bos_token = '<|im_start|>'
        tokenizer.eos_token = '<|im_end|>'
        tokenizer.pad_token_id = tokenizer.eod_id
    elif tokenizer.pad_token_id is None:
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None \
            else tokenizer.eos_token_id

    # merge lora weights and save hf model    
    model = PeftModel.from_pretrained(model, args.peft_model, device_map=device_map)
    model = model.merge_and_unload()
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

if __name__ == "__main__":
    args = parse_args()
    merge_and_save(args)

