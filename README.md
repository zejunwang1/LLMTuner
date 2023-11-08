# LLMTuner: 大语言模型指令调优工具

## 项目介绍

LLMTuner 是一个支持 LoRA、QLoRA 和全量参数微调的大语言模型指令调优工具。在训练中可以使用 **flash attention**、**xformers attention** 技术提升训练效率，并结合 **LoRA**、 **DeepSpeed ZeRO**、**gradient checkpointing** 和 **4-bit 量化**等技术，有效降低显存占用，实现在单张消费级显卡 (A100/A40/A30/RTX3090/V100) 上微调 7B/13B/34B 大模型。

目前支持进行微调的基座模型：

- Aquila & Aquila2 (`BAAI/Aquila-7B`, `BAAI/Aquila2-7B`, `BAAI/Aquila2-34B`, etc.)

- Baichuan & Baichuan 2 (`baichuan-inc/Baichuan-7B`, `baichuan-inc/Baichuan-13B-Base`, `baichuan-inc/Baichuan2-7B-Base`, `baichuan-inc/Baichuan2-13B-Base`, etc.)

- BLOOM & BLOOMZ (`bigscience/bloom`, `bigscience/bloomz`, etc.)

- InternLM (`internlm/internlm-7b`, `internlm/internlm-20b`, etc.)

- LLaMA & LLaMA 2 (`meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`, etc.)

- Qwen (`Qwen/Qwen-7B`, `Qwen/Qwen-14B`, etc.)

## 训练环境

- torch == 2.0.1

- tiktoken == 0.5.1

- transformers == 4.32.0

- sentencepiece == 0.1.99

- bitsandbytes == 0.41.1

- peft == 0.5.0

- deepspeed == 0.11.1

- xformers == 0.0.22

- tensorboardX

推荐安装 [flash attention](https://github.com/Dao-AILab/flash-attention) 提高训练效率：

```shell
# Make sure that ninja is installed
pip install ninja
# Compile from source
git clone --recursive https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install
# Install dropout_layer_norm and rotary-emb
pip install csrc/layer_norm
pip install csrc/rotary
```

## 训练数据

### 指令数据

指令微调的训练数据为 jsonl 格式，每一行的数据形式如下，其中 conversation 字段是必需的，可参考 data/dummy.jsonl 文件。

```json
{
    "conversation_id": 4, 
    "category": "Brainstorming", 
    "num_turns": 4, 
    "conversation": [
        {
            "human": "如何制定一个有效的网页设计？", 
            "assistant": "\n\n要制定一个有效的网页设计，需要考虑以下几个方面：\n\n1. 目标受众：确定您的目标受众是谁，这将有助于您决定网站的设计和内容。\n\n2. 清晰的布局：网站的布局应该简单，易于使用和导航。所有元素应该有明确的标识和位置，以便用户快速找到他们需要的信息。\n\n3. 吸引人的视觉效果：网站的设计应该吸引人，使用清晰的图像、颜色和字体。但同时也要确保视觉效果不会影响网站的性能和导航。\n\n4. 有效的内容：网站的内容应该有价值，易于理解和使用。使用简单的语言和段落，避免使用过于复杂的术语和语言。\n\n5. 可访问性：网站应该易于访问，即使是对于有视觉或听力障碍的用户。确保网站符合标准的可访问性要求。\n\n6. 响应式设计：网站应该是响应式的，即适合各种不同尺寸的设备。这将确保用户在不同设备上都能够访问和使用您的网站。\n\n通过考虑这些方面，您可以制定一个有效的网页设计，吸引并留住您的受众，并提高您的网站的成功率。"
        }, 
        {
            "human": "哪些因素会影响网站的性能？", 
            "assistant": "网站的性能受许多因素的影响，以下是一些主要的因素：\n\n1. 服务器性能：网站的速度和性能取决于服务器的性能，包括硬件、软件、配置和网络连接速度。\n\n2. 图片和媒体文件的大小：大尺寸的图片和媒体文件会增加网站的加载时间，因此需要使用适当的图片和媒体文件大小，以确保网站可以快速加载。\n\n3. 代码优化：网站的代码应尽可能简洁和优化，以确保它在加载和运行时的速度最快。\n\n4. 缓存和压缩：使用缓存和压缩技术可以大大提高网站的性能，因为它们可以减少文件大小和加载时间。\n\n5. 第三方插件和脚本：第三方插件和脚本可能会减慢网站的速度，因此需要谨慎选择并仅使用必要的插件和脚本。\n\n6. 浏览器兼容性：网站应该在各种不同的浏览器上兼容，因为不同的浏览器可能会以不同的方式解释代码并表现出不同的性能。\n\n7. 网络连接速度：用户的网络连接速度也会影响网站的性能，如果用户的网络连接速度较慢，网站的加载时间可能会更长。\n\n总之，要提高网站的性能，需要优化服务器、代码和媒体文件，使用缓存和压缩技术，谨慎使用第三方插件和脚本，并考虑用户的网络连接速度和浏览器兼容性。"
        }, 
        {
            "human": "如何确定网站的目标受众？", 
            "assistant": "确定网站的目标受众是成功的关键之一，以下是确定目标受众的一些步骤：\n\n1. 考虑您的业务：首先，您需要考虑您的业务或服务类型，以及您想要吸引的客户或用户类型。比如，如果您是一家专门从事儿童玩具销售的公司，您的目标受众可能是父母或祖父母。\n\n2. 调查市场：进行市场调查，了解您的目标受众的需求、喜好、行为和偏好。您可以使用各种市场调查技术，如在线调查、焦点小组讨论等等。\n\n3. 竞争分析：研究竞争对手的网站，了解他们的目标受众、品牌声誉和营销策略。\n\n4. 客户反馈：收集客户反馈，了解他们对您网站的看法，以及您的网站是否满足他们的需求和期望。\n\n5. 利用分析工具：使用分析工具（如Google Analytics）了解您的网站的访问者特征，比如年龄、性别、地理位置等，从而了解您的目标受众。\n\n综上所述，确定网站的目标受众需要考虑您的业务类型、市场调查、竞争分析、客户反馈和分析工具。这些步骤可以帮助您更好地了解您的目标受众，从而制定更好的网站设计和营销策略。"
        }, 
        {
            "human": "如何在网站上提供更好的用户体验？", 
            "assistant": "提供更好的用户体验是一个成功的网站设计的关键。以下是一些提高网站用户体验的方法：\n\n1. 清晰的页面布局：网站应该有清晰的页面布局，易于使用和导航。所有元素应该有明确的标识和位置，以便用户快速找到他们需要的信息。\n\n2. 易于理解和使用的导航：网站的导航应该简单、明确和易于理解。用户应该能够快速找到他们需要的信息，而不必浪费时间在网站上寻找。\n\n3. 快速的网站加载速度：网站的加载速度是提供良好用户体验的关键因素之一。网站应该尽可能快地加载，以避免用户等待过长的时间。\n\n4. 响应式设计：网站应该是响应式的，即适合各种不同尺寸的设备。这将确保用户在不同设备上都能够访问和使用您的网站。\n\n5. 良好的内容：网站的内容应该有价值，易于理解和使用。使用简单的语言和段落，避免使用过于复杂的术语和语言。\n\n6. 与用户互动：网站应该与用户互动，以吸引和留住用户。例如，您可以添加评论部分、交互式表单和调查等功能。\n\n7. 提供搜索功能：对于较大的网站，提供搜索功能可以帮助用户快速找到他们需要的信息。\n\n8. 避免繁琐的注册流程：网站应该尽可能避免繁琐的注册流程，以避免阻碍用户的使用。\n\n9. 优化网站的可访问性：网站应该易于访问，即使是对于有视觉或听力障碍的用户。确保网站符合标准的可访问性要求。\n\n总之，提供更好的用户体验需要考虑页面布局、导航、加载速度、响应式设计、内容、互动、搜索功能、注册流程和可访问性。通过优化这些方面，您可以提高用户满意度，增加用户留存率，并提高您的网站的成功率。"
        }
    ]
}
```

我们将 [MOSS](https://github.com/OpenLMLab/MOSS) 项目开源的 [moss-003-sft-data](https://huggingface.co/datasets/fnlp/moss-003-sft-data) 多轮对话数据处理成了如上统一的数据格式，在本项目的微调中可以直接使用。该数据集基于`gpt-3.5-turbo`构造而成，具有更高的数据质量和更长的对话轮数，约含110万条对话数据。

| dataset                                                                                  | 描述                                      |
| ---------------------------------------------------------------------------------------- | --------------------------------------- |
| [WangZeJun/moss-003-sft-all](https://huggingface.co/datasets/WangZeJun/moss-003-sft-all) | 将近110万条多轮对话数据，基于 moss-003-sft-data 处理得到 |
| [WangZeJun/moss-003-sft-30w](https://huggingface.co/datasets/WangZeJun/moss-003-sft-30w) | 约30万条多轮对话数据，根据 moss-003-sft-data 采样得到   |
| [WangZeJun/moss-003-sft-21w](https://huggingface.co/datasets/WangZeJun/moss-003-sft-21w) | 约21万条多轮对话数据，根据 moss-003-sft-data 采样得到   |

对于指令微调，LLMTuner 按照如下格式对多轮对话数据进行 tokenize

```context
<s>{human}</s><s>{assistant}</s><s>{human}</s><s>{assistant}</s><s>{human}</s><s>{assistant}</s>
```

### 特定任务数据

以广告文案生成任务为例，微调的训练数据需为 jsonl 格式，每一行必须包含 source 和 target 两个字段，可参考 data/task_dummy.jsonl 文件。

```json
{
    "source": "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤", 
    "target": "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。毕竟好穿时尚，谁都能穿出腿长2米的效果宽松的裤腿，当然是遮肉小能手啊。上身随性自然不拘束，面料亲肤舒适贴身体验感棒棒哒。系带部分增加设计看点，还让单品的设计感更强。腿部线条若隐若现的，性感撩人。颜色敲温柔的，与裤子本身所呈现的风格有点反差萌。"
}
```

对于特定任务微调，LLMTuner 按照如下格式对输入数据进行 tokenize

```context
<s>{prompt}{source}</s><s>{target}</s>
```

## 模型训练

### QLoRA

[QLoRA](https://github.com/artidoro/qlora) 是一种高效的微调方法，可以在保持完整的16位微调任务性能下，实现单个 48GB GPU 上微调 65B 参数量模型。

使用 QLoRA 微调`Baichuan2-7B-Base`模型：

```shell
python tuner/train_qlora.py \
    --model_type baichuan \
    --model_name_or_path /path/to/Baichuan2-7B-Base/ \
    --data_path data/dummy.jsonl \
    --output_dir dummy_output \
    --max_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 0.3 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --bf16 True \
    --tf32 True \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --optim "paged_adamw_32bit" \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bits 4 \
    --double_quant True \
    --quant_type "nf4"
```

开启 flash attention，微调`Llama-2-7b`或`Qwen-7B`模型：

```shell
python tuner/train_qlora.py \
    --model_type llama \
    --model_name_or_path /path/to/Llama-2-7b/ \
    --data_path data/dummy.jsonl \
    --output_dir dummy_output \
    --max_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --use_flash_attn True \
    --use_xformers_attn False \
    --max_grad_norm 0.3 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --bf16 True \
    --tf32 True \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --optim "paged_adamw_32bit" \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bits 4 \
    --double_quant True \
    --quant_type "nf4"
```

若手中有多张 GPU 卡，可通过 torchrun 开启单机多卡训练提升效率。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20001 tuner/train_qlora.py
```

### LoRA

[LoRA](https://github.com/microsoft/LoRA) 的核心思想是冻结预训练模型权重，将可训练的秩分解矩阵注入 Transformer 架构 (attention 部分) 的每一层，从而大大减少了下游任务的微调参数量。

使用 LoRA 微调不同基座模型，对应的`target_modules`参考设置如下：

| 模型                    | model_type | target_modules  |
| --------------------- | ---------- | --------------- |
| LLaMA & LLaMA 2       | llama      | q_proj,v_proj   |
| Baichuan & Baichuan 2 | baichuan   | W_pack          |
| BLOOM & BLOOMZ        | bloom      | query_key_value |
| InternLM              | intern     | q_proj,v_proj   |
| Qwen                  | qwen       | c_attn          |

实际微调时，可根据基座模型类型增减`target_modules`。

使用 LoRA 微调`Llama-2-7b`模型：

```shell
python tuner/train_lora.py \
    --model_type llama \
    --model_name_or_path /path/to/Llama-2-7b/ \
    --target_modules "q_proj,k_proj,v_proj,o_proj" \
    --data_path data/dummy.jsonl \
    --output_dir dummy_output \
    --max_length 2048 \
    --use_flash_attn True \
    --use_xformers_attn False \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 1.0 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --bf16 True \
    --tf32 True \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --optim "paged_adamw_32bit" \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bits 16 \
    --additional_trainable_params "None"
```

通过设置`--additional_trainable_params`选项可增加微调过程中的可训练参数。

### 全量参数微调

本项目支持使用 [DeepSpeed Zero](https://www.deepspeed.ai/training/) 进行大模型全量参数的微调。全量参数微调`bigscience/bloomz-1b7`模型：

```shell
deepspeed --include localhost:0 tuner/train_full.py \
    --model_type bloom \
    --model_name_or_path /path/to/bloomz-1b7/ \
    --data_path data/dummy.jsonl \
    --output_dir dummy_output \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 1.0 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --bf16 True \
    --tf32 True \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed data/ds_config_zero2.json
```

全量参数微调 [TinyLlama-1.1B](https://github.com/jzhang38/TinyLlama) 模型：

```shell
deepspeed --include localhost:0 tuner/train_full.py \
    --model_type llama \
    --model_name_or_path /path/to/TinyLlama-1.1B-intermediate-step-480k-1T/ \
    --data_path data/dummy.jsonl \
    --output_dir dummy_output \
    --max_length 1024 \
    --use_flash_attn True \
    --use_xformers_attn False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 1.0 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --bf16 True \
    --tf32 True \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed data/ds_config_zero2.json
```

若手中有多张 GPU 卡，可开启单机多卡训练提升效率。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
deepspeed --include localhost:0,1,2,3 tuner/train_full.py
```

### 特定任务微调

使用 QLoRA 在广告文案生成任务上微调`Baichuan2-7B-Base`模型，设置`task_finetune=True`和`task_prompt="广告营销文案生成:\n"`。

```shell
python tuner/train_qlora.py \
    --model_type baichuan \
    --model_name_or_path /path/to/Baichuan2-7B-Base/ \
    --data_path data/task_dummy.jsonl \
    --output_dir dummy_output \
    --max_length 1024 \
    --task_finetune True \
    --task_prompt "广告营销文案生成:\n" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 0.3 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_ratio 0.03 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --bf16 True \
    --tf32 True \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --optim "paged_adamw_32bit" \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bits 4 \
    --double_quant True \
    --quant_type "nf4"
```

## 模型推理

### 合并 LoRA 权重

```shell
python inference/merge_lora_weights_and_save.py
usage: merge_lora_weights_and_save.py [-h] --base_model BASE_MODEL --peft_model PEFT_MODEL --save_dir SAVE_DIR
                                      [--cache_dir CACHE_DIR] [--device {cpu,cuda}]
```

### 代码调用

可通过如下代码调用本项目微调后的模型来生成对话

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/path/to/model", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

query = "晚上睡不着怎么办"
input_ids = [tokenizer.bos_token_id]
input_ids.extend(tokenizer.encode(query))
input_ids.append(tokenizer.eos_token_id)
input_ids.append(tokenizer.bos_token_id)
input_ids = torch.tensor([input_ids], device=model.device)

outputs = model.generate(input_ids, do_sample=True, top_p=0.85, top_k=8, temperature=0.3, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.1)
response_ids = outputs[0][len(input_ids[0]): ]
response = tokenizer.decode(response_ids, skip_special_tokens=True)
print(response)
```

### 命令行工具方式

```shell
python inference/cli_demo.py
```

## 引用

若使用本项目的数据或代码，请引用本项目。

```
@misc{LLMTuner,
  author = {Zejun Wang},
  title = {LLMTuner: 大语言模型指令调优工具},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zejunwang1/LLMTuner}}
}
```

## ⭐️ Star History

![Star History Chart](https://api.star-history.com/svg?repos=zejunwang1/LLMTuner&type=Date)
