import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# ------------------------------------------------------------------------------------
# 步骤 1: 定义所有超参数和配置
# ------------------------------------------------------------------------------------

# 微调模型的ID
model_id = "microsoft/Phi-3-mini-4k-instruct"

# 数据集文件路径
dataset_file = "dataset.jsonl"

# LoRA 配置
lora_r = 16                # LoRA的秩
lora_alpha = 32            # LoRA的alpha参数
lora_dropout = 0.05        # LoRA的dropout率

# BitsAndBytes 量化配置
use_4bit = True            # 是否使用4-bit量化
bnb_4bit_compute_dtype = "bfloat16" # 4-bit计算时使用的数据类型
bnb_4bit_quant_type = "nf4" # 量化类型
use_nested_quant = False   # 是否使用嵌套量化

# 训练参数
output_dir = "./code_poet_model"  # 训练输出目录
num_train_epochs = 3              # 训练轮数
per_device_train_batch_size = 1   # 每个设备的训练批量大小
gradient_accumulation_steps = 4   # 梯度累积步数，等效批量大小为 batch_size * grad_acc_steps
gradient_checkpointing = True     # 是否使用梯度检查点
max_grad_norm = 0.3               # 最大梯度范数
learning_rate = 2e-4              # 学习率
weight_decay = 0.001              # 权重衰减
optim = "paged_adamw_8bit"        # 优化器类型
lr_scheduler_type = "cosine"      # 学习率调度器类型
max_steps = -1                    # 最大训练步数（-1表示由epoch决定）
warmup_ratio = 0.03               # 预热比例
group_by_length = True            # 是否按长度对序列进行分组
save_steps = 25                   # 每25步保存一次检查点
logging_steps = 5                 # 每5步记录一次日志

# SFT Trainer 参数
max_seq_length = 1024             # 最大序列长度
packing = False                   # 是否使用打包技术
device_map = {"": 0}               # 指定模型加载到哪个GPU上

# ------------------------------------------------------------------------------------
# 步骤 2: 加载数据集
# ------------------------------------------------------------------------------------
print("正在加载数据集...")
# 我们需要将jsonl的每一行转换成对话格式
def format_instruction(sample):
    # Phi-3的指令格式是 <|user|>\n{prompt}<|end|>\n<|assistant|>\n{response}<|end|>
	return f"<|user|>\n{sample['prompt']}<|end|>\n<|assistant|>\n{sample['response']}<|end|>"

dataset = load_dataset("json", data_files=dataset_file, split="train")
# 对整个数据集应用这个格式化函数
formatted_dataset = dataset.map(lambda x: {"text": format_instruction(x)})
print("数据集加载并格式化完毕。")

# ------------------------------------------------------------------------------------
# 步骤 3: 加载模型和分词器
# ------------------------------------------------------------------------------------
print("正在加载模型和分词器...")
# 配置BitsAndBytes，这是实现4-bit量化的关键
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True # Phi-3需要这个参数
)
model.config.use_cache = False # 在训练时禁用缓存
model.config.pretraining_tp = 1

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Phi-3的tokenizer没有默认的pad_token，我们用eos_token来代替
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("模型和分词器加载完毕。")

# ------------------------------------------------------------------------------------
# 步骤 4: 配置LoRA (PEFT)
# ------------------------------------------------------------------------------------
print("正在配置LoRA...")
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    # 对于Phi-3, 建议的目标模块
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        "gate_proj",
        "down_proj",
        "up_proj",
    ]
)
print("LoRA配置完毕。")

# ------------------------------------------------------------------------------------
# 步骤 5: 配置训练参数
# ------------------------------------------------------------------------------------
print("正在配置训练参数...")
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=False, 
    bf16=True,  
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    # max_seq_length=max_seq_length # <--- 把参数加到这里
)
print("训练参数配置完毕。")

# ------------------------------------------------------------------------------------
# 步骤 6: 初始化并开始训练
# ------------------------------------------------------------------------------------
print("Initializing SFT Trainer...")
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    dataset_text_field="text", # This should work with the latest TRL
    max_seq_length=1024,
    packing=False,
	processing_class=tokenizer
	
)

print("训练开始！")
trainer.train()

# 保存最终的模型
trainer.save_model(output_dir)
print(f"训练完成，模型已保存至 {output_dir}")