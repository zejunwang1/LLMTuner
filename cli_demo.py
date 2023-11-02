# coding=utf-8

import argparse
import torch
import transformers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--multi_round", action="store_true")
    parser.add_argument("--history_max_tokens", type=int, default=2048)
    args = parser.parse_args()
    return args

def main(args):
    # load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    print("clear清空历史对话, quit/stop退出")
    history = []
    if args.multi_round:
        while True:
            text = input("User: ")
            if text == "stop" or text == "quit":
                break
            if text == "clear":
                history = []
                continue
            history += tokenizer(text).input_ids
            history.append(tokenizer.eos_token_id)
            history = history[-args.history_max_tokens:]
            input_ids = torch.tensor([history], device=model.device)
            outputs = model.generate(
                input_ids,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty
            )
            
            input_ids_len = input_ids.size(1)
            response_ids = outputs[0][input_ids_len: ]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            print("Assistant: {}\n".format(response))
            
            history += response_ids.tolist()
            if history[-1] != tokenizer.eos_token_id:
                history.append(tokenizer.eos_token_id)
    else:
        while True:
            text = input("User: ")
            if text == "stop" or text == "quit":
                break
            input_ids = tokenizer(text).input_ids
            input_ids.append(tokenizer.eos_token_id)
            input_ids = torch.tensor([input_ids], device=model.device)
            outputs = model.generate(
                input_ids,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty
            )
            
            input_ids_len = input_ids.size(1)
            response_ids = outputs[0][input_ids_len: ]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            print("Assistant: {}\n".format(response))

if __name__ == "__main__":
    args = parse_args()
    main(args)

