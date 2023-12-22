# coding=utf-8

import argparse
import torch
import transformers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--task_prompt", type=str, default=None)
    parser.add_argument("--multi_round", action="store_true")
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
    prompt = args.task_prompt + "{}" if args.task_prompt is not None else None
    history = []
    if args.multi_round:
        while True:
            text = input("User: ")
            if text == "stop" or text == "quit":
                break
            if text == "clear":
                history = []
                continue
            input_ids = [tokenizer.bos_token_id]
            input_ids.extend(tokenizer.encode(text))
            input_ids.append(tokenizer.eos_token_id)
            input_ids.append(tokenizer.bos_token_id)
            input_ids = torch.tensor([input_ids], device=model.device)
            history = input_ids if len(history) == 0 else torch.concat((history, input_ids), dim=-1)
            # truncate left
            history = history[:, -args.max_input_tokens: ]

            outputs = model.generate(
                history,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty
            )
            
            response_ids = outputs[0][len(history[0]): ]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            print("Assistant: {}\n".format(response))
            
            history = torch.concat((history, outputs[:, len(history[0]): ]), dim=-1)
    else:
        while True:
            text = input("User: ")
            if text == "stop" or text == "quit":
                break
            input_ids = [tokenizer.bos_token_id]
            input_ids.extend(tokenizer.encode(prompt.format(text) if prompt is not None else text))
            input_ids.append(tokenizer.eos_token_id)
            input_ids.append(tokenizer.bos_token_id)
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
            
            response_ids = outputs[0][len(input_ids[0]): ]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            print("Assistant: {}\n".format(response))

if __name__ == "__main__":
    args = parse_args()
    main(args)

