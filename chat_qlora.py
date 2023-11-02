# coding=utf-8

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        #load_in_4bit=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter_name_or_path)
    model.to(device)
    model.eval()

    print("基于bloom-qlora的聊天机器人, clear清空历史对话, quit/stop退出")
    input_pattern = "{}</s>"
    history = []
    if args.multi_round:
        while True:
            text = input("用户: ")
            if text == "stop" or text == "quit":
                break
            if text == "clear":
                history = []
                continue
            history += tokenizer(input_pattern.format(text)).input_ids
            history = history[-args.history_max_tokens:]
            input_ids = torch.tensor([history], device=device)
            outputs = model.generate(
                input_ids=input_ids,
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
            response = tokenizer.decode(response_ids)
            print("Assistant: {}\n".format(response.strip().replace("</s>", "")))

            history += response_ids.tolist()
            if history[-1] != tokenizer.eos_token_id:
                history.append(tokenizer.eos_token_id)
    else:
        while True:
            text = input("用户: ")
            if text == "stop" or text == "quit":
                break
            input_ids = tokenizer(input_pattern.format(text), return_tensors="pt").input_ids
            input_ids = input_ids.to(device)
            outputs = model.generate(
                input_ids=input_ids,
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
            response = tokenizer.decode(response_ids)
            print("Assistant: {}\n".format(response.strip().replace("</s>", "")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--adapter_name_or_path", required=True, type=str)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--multi_round", action="store_true")
    parser.add_argument("--history_max_tokens", type=int, default=1024)
    args = parser.parse_args()
    main(args)

