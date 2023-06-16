# LLMTuner: 大语言模型指令调优工具（支持全量参数微调、LoRA 和 QLoRA）

## News

- 开源在 12w 条多轮对话数据上全量参数微调得到的 [bloom-3b-moss-chat]() 

- 使用 QLoRA 技术，在 12w 条多轮对话数据上微调得到的 [bloomz-7b1-qlora-moss-chat]()  

- 开源全量参数训练、LoRA 和 QLoRA 训练代码。

## 项目简介

LLMTuner 是一个支持全量参数微调、LoRA 和 QLoRA 的大语言模型指令调优工具。在训练中可以结合词表裁剪、DeepSpeed-ZeRO、gradient checkpointing、4-bit 量化等技术，有效降低显存占用，实现在单张消费级显卡上微调 7B/13B/33B 大模型。

我们从 MOSS 项目开源的中英文指令微调数据集 [moss-003-sft-data](https://huggingface.co/datasets/fnlp/moss-003-sft-data) 中抽取 12w 条多轮对话数据作为训练数据，分别：

- 以 [词表裁剪后的 bloom-3b](https://huggingface.co/YeungNLP/bloom-2b6-zh) 为基座，全量参数微调得到 [bloom-3b-moss-chat]()

- 以 [词表裁剪后的 bloomz-7b1-mt](https://huggingface.co/YeungNLP/bloomz-6b4-mt-zh) 为基座，使用 QLoRA 技术微调得到 [bloomz-7b1-qlora-moss-chat]()

## Requirements

- bitsandbytes==0.39.0

- transformers @ git+https://github.com/huggingface/transformers.git

- peft @ git+https://github.com/huggingface/peft.git

- accelerate @ git+https://github.com/huggingface/accelerate.git

- tensorboardX

```shell
pip install -U -r requirements.txt
```

## 训练数据

训练数据中的单条指令需要预处理为如下形式：

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

## 全量参数微调

基于 DeepSpeed ZeRO Stage 3 的单卡训练：

```shell
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
    --warmup_steps 500 \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --report_to "tensorboard" \
    --bf16 True \
    --tf32 True \
    --deepspeed data/deepspeed.json
```

设置 `max_length=1024, batch_size=16, bf16=True`，单卡需要约 45G 显存对词表裁剪后的 bloom-3b 基座进行全量参数微调，在 12w 多轮对话数据集上训练了一个 epoch（大约 8000 steps），训练过程中的 loss 变化如下：

<img src="images/ds_loss.png" width="500">

## LoRA

[LoRA](https://github.com/microsoft/LoRA) 的核心思想是冻结预训练模型权重，将可训练的秩分解矩阵注入 Transformer 架构的每一层，从而大大减少了下游任务的微调参数量。

<img src="images/lora1.png" width="250">

LoRA 的实现流程概述如下：

- 在原始预训练语言模型 (PLM) 旁增加一个旁路，做一个先降维再升维的操作，以此来模拟所谓的本征秩 (intrinsic rank)；

- 训练的时候固定 PLM 的参数不变，只训练降维矩阵 $A$ 和升维矩阵 $B$，即优化器只优化右路的参数；

- 模型的输入、输出维度不变，左右两边共用模型的输入，输出时将 PLM 与旁路的输出叠加：$h=Wx+BAx$

- 用随机高斯分布 $N(0,\sigma^2)$ 初始化 $A$，用全零矩阵初始化 $B$。矩阵 $B$ 的全零初始化，使得在训练最开始的一段时间，右路的结果会接近于0，这样模块的输出就基本上来自于左路，也就是大模型原有参数的计算结果，这使得模型优化的初始点和原始的大模型保持一致。

使用 LoRA 进行单卡训练：

```shell
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
    --warmup_steps 500 \
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
```

```
trainable params: 31,457,280 || all params: 6,261,878,784 || trainable%: 0.5023616886417199
```

## QLoRA

[QLoRA](https://github.com/artidoro/qlora) 是一种高效的微调方法，可以在保持完整的16位微调任务性能下，实现单个 48GB GPU 上微调 65B 参数量模型。QLoRA 通过冻结的 4-bit 量化预训练语言模型向低秩适配器(LoRA) 反向传播梯度。使用 4-bit NormalFloat (NF4) 量化、Double Quantization、Paged Optimizers、所有 Linear 层插入 adapter 等技术，QLoRA 在不牺牲性能的情况下大大节省了显存占用。

<img src="images/qlora.png" width="600">

使用 QLoRA 进行单卡训练：

```shell
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
    --warmup_steps 500 \
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
```

以词表裁剪后的 bloomz-7b1-mt 为基座，在上面参数配置下，单卡仅需要约 12G 显存便可进行 QLoRA 微调。由于 QLoRA 在所有的全连接层处都插入了 Adapter，相同 `lora_r=64` 下其可训练的参数量多于 LoRA：

```
trainable params: 125,829,120 || all params: 3,336,351,744 || trainable%: 3.7714584568694685
```

在 12w 多轮对话数据集上训练了一个 epoch（大约 8000 steps），训练过程中的 loss 变化如下：

<img src="images/qlora_loss.png" width="500">

## 生成效果

#### bloom-3b-moss-chat

命令行单轮对话：

```shell
python chat.py \
    --model_name_or_path /path/to/model \
    --top_p 0.85 \
    --temperature 0.3 \
    --max_new_tokens 512 \
    --repetition_penalty 1.2
```

添加 `--multi_round` 可开启多轮对话。

<details><summary><b>单轮对话</b></summary>

| 输入                                                                                                                                                                | 输出                                                                                                                                          |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 如何成为一个全栈工程师？                                                                                                                             | 成为一名全栈工程师需要具备以下技能和知识：
1. 编程语言基础：熟练掌握至少一种编程语言，如Java、Python等。
2. 数据结构和算法：了解常见的数据结构（如数组、链表等）以及算法的基本原理和使用方法。
3. 网络通信与Web开发：熟悉HTTP协议、TCP/IP协议、网络安全等方面的知识和应用场景。
4. 前端技术：学习HTML/CSS/JavaScript等前端技术的实现方法和常用框架（如React、Vue.js）。
5. 后端技术：熟悉常见的后端技术和框架（如Node.js、Django、Ruby on Rails）等的使用方法和性能优化技巧。
6. 项目管理能力：在项目中能够有效地组织和协调团队成员的工作，并负责项目的进度和资源分配等工作。
7. 持续学习的能力：保持对新技术和工具的敏感度和好奇心，不断更新自己的知识和技能。 |
| 以咏梅为题目，帮我写一首古诗，要求包含梅花、白雪等元素。古诗：                                                                                                                                   | 白雪纷纷落，<br>梅花含笑开。<br>清香扑鼻来，<br>素雅动人心。<br>枝头挂冰玉，<br>叶上缀珠玑。<br>寒冬不凋零，<br>春风又何妨？<br>此情谁能识，<br>惟愿君知晓。                                            |
|我要与朋友分别了，请你写一首古诗，表达我的离别之情。|好的，我将为您创作一首关于离别的诗歌。<br>离别情深似海潮，<br>相思难耐泪涟涟。<br>天涯何处觅知音？<br>唯有明月共清辉|
| 背诵苏轼的《水调歌头·明月几时》                                                                                                                                                  | 明月几时有，把酒问青天。不知天上宫阙，今夕是何年？我欲乘风归去，又恐琼楼玉宇，高处不胜寒；起舞弄清影，何似在人间！转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。                          |

</details>


#### bloomz-7b1-qlora-moss-chat

命令行单轮对话：

```shell
python chat_qlora.py \
    --model_name_or_path /path/to/model \
    --adapter_name_or_path /path/to/adapter \
    --top_p 0.85 \
    --temperature 0.3 \
    --max_new_tokens 512 \
    --repetition_penalty 1.2 \
    --history_max_tokens 1024
```

添加 `--multi_round` 可开启多轮对话。需要注意的是，bloomz-7b1-qlora-moss-chat 的基座为词表裁剪后的 bloomz-7b1-mt 模型。



## 参考链接

- [Firefly(流萤): 中文对话式大语言模型](https://github.com/yangjianxin1/Firefly)

- [LLMPruner：大语言模型裁剪工具](https://github.com/yangjianxin1/LLMPruner) 

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://github.com/artidoro/qlora) 

- [🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning.](https://github.com/huggingface/peft)

- [MOSS: An open-source tool-augmented conversational language model from Fudan University](https://github.com/OpenLMLab/MOSS) 

## 引用

若使用本项目的代码或模型，请引用本项目。

```
@misc{LLMTuner,
  author = {Zejun Wang},
  title = {LLMTuner: 大语言模型指令调优工具(支持全量参数微调、LoRA 和 QLoRA)},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zejunwang1/LLMTuner}}
}
```
