
# full finetune
python train.py \
	--model_name_or_path /path/to/bloom \
	--data_path data/train.jsonl \
	--output_dir output/bloom-3b-moss-chat \
	--max_length 1024 \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 16 \
	--learning_rate 2e-5 \
	--num_train_epochs 1 \
	--lr_scheduler_type "cosine" \
	--warmup_ratio 0.05 \
	--logging_steps 10 \
	--save_strategy "steps" \
	--save_steps 200 \
	--save_total_limit 1 \
	--report_to "tensorboard" \
	--bf16 True \
	--tf32 True

# full finetune with DDP
torchrun --nproc_per_node=4 train.py \
	--model_name_or_path /path/to/bloom \
        --data_path data/train.jsonl \
        --output_dir output/bloom-3b-moss-chat \
        --max_length 1024 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.05 \
        --logging_steps 10 \
        --save_strategy "steps" \
        --save_steps 200 \
        --save_total_limit 1 \
        --report_to "tensorboard" \
        --bf16 True \
        --tf32 True

# full finetune with deepspeed
deepspeed --include localhost:0 train.py \
	--model_name_or_path /path/to/bloom \
	--data_path data/train.jsonl \
	--output_dir output/bloom-3b-moss-chat/ \
	--max_length 1024 \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 4 \
	--learning_rate 2e-5 \
	--num_train_epochs 1 \
	--lr_scheduler_type "cosine" \
	--warmup_ratio 0.05 \
	--logging_steps 10 \
	--save_strategy "steps" \
	--save_steps 500 \
	--save_total_limit 1 \
	--report_to "tensorboard" \
	--bf16 True \
	--tf32 True \
	--deepspeed data/deepspeed.json

# LoRA
python train_lora.py \
	--model_type bloom
	--model_name_or_path /path/to/bloom \
	--data_path data/train.jsonl \
	--output_dir output/bloomz-7b1-lora-moss-chat \
	--max_length 1024 \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 4 \
	--max_grad_norm 0.3 \
	--learning_rate 2e-4 \
	--num_train_epochs 1 \
	--lr_scheduler_type "cosine" \
	--warmup_ratio 0.05 \
	--logging_steps 10 \
	--save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 1 \
	--report_to "tensorboard" \
	--gradient_checkpointing True \
	--lora_r 64 \
	--lora_alpha 16 \
	--lora_dropout 0.05 \
	--int8_training True

# QLoRA
python train_qlora.py \
	--model_name_or_path /path/to/bloom \
	--data_path data/train.jsonl \
	--output_dir output/bloomz-7b1-qlora-moss-chat \
	--max_length 1024 \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 4 \
	--max_grad_norm 0.3 \
	--learning_rate 2e-4 \
	--num_train_epochs 1 \
	--lr_scheduler_type "cosine" \
	--warmup_ratio 0.05 \
       	--logging_steps 10 \
	--save_strategy "steps" \
	--save_steps 500 \
       	--save_total_limit 1 \
	--report_to "tensorboard" \
	--bf16 True \
	--tf32 True \
	--gradient_checkpointing True \
	--optim "paged_adamw_32bit" \
	--lora_r 64 \
	--lora_alpha 16 \
        --lora_dropout 0.05 \
	--bits 4 \
	--double_quant True \
	--quant_type "nf4"

