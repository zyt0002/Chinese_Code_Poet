"""
中文诗人模型训练脚本

这个脚本使用LoRA（低秩适应）技术对Phi-3模型进行微调，
训练一个能够生成中文诗歌的AI助手。

主要步骤：
1. 加载训练数据（问答对格式）
2. 加载预训练的Phi-3模型和分词器
3. 配置LoRA参数（参数高效微调）
4. 设置训练参数
5. 开始训练并保存模型

关键概念：
- LoRA: 只训练少量新增参数，而不修改原模型
- 量化: 将模型压缩到4位精度，节省显存
- SFT: 监督微调，使用问答对数据训练模型
"""

# 导入必要的库
import torch  # PyTorch深度学习框架
from datasets import load_dataset  # Hugging Face数据集库，用于加载训练数据
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # Transformers库，用于加载预训练模型
from peft import LoraConfig  # PEFT库，用于参数高效微调
from trl import SFTConfig, SFTTrainer  # TRL库，用于监督微调
import os  # 操作系统接口，用于文件和目录操作
# --- 1. 配置参数 ---
# 基础模型ID，这是微软发布的Phi-3小型模型，专门用于指令跟随
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
# 训练数据文件名，包含问答对的JSON Lines格式文件
DATASET_FILE = "ai_studio_code.jsonl"
# 输出目录，训练完成后模型会保存在这里
OUTPUT_DIR = "./code_poet_final_guaranteed"
# 创建输出目录，如果目录已存在则不报错
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 加载数据集 ---
print("Step 1: Loading dataset...")
# 从JSON Lines文件加载数据集，split="train"表示这是训练数据
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

# 格式化数据集以适应SFT（监督微调）训练 - 使用prompt/completion格式
def format_prompt(example):
    """
    将原始数据转换为训练所需的格式
    输入: example包含'prompt'和'response'字段
    输出: 包含'prompt'和'completion'字段的字典
    """
    # SFTTrainer期望有prompt和completion字段
    # 将用户输入格式化为对话格式
    formatted_prompt = f"用户: {example['prompt']}\n助手:"
    # 模型应该生成的回答，前面加一个空格
    completion = f" {example['response']}"
    return {
        "prompt": formatted_prompt,  # 输入提示
        "completion": completion     # 期望的输出
    }

# 对数据集中的每个样本应用格式化函数
dataset = dataset.map(format_prompt)
print("Dataset loaded and formatted.")

# --- 3. 加载模型和分词器 ---
print("Step 2: Loading base model and tokenizer...")

# 定义4位量化配置，用于减少显存占用
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用4位量化，将模型权重从32位压缩到4位
    bnb_4bit_quant_type="nf4",  # 使用NF4量化类型，这是一种优化的量化方法
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用bfloat16精度
    bnb_4bit_use_double_quant=True,  # 启用双重量化，进一步节省内存
)

# 加载预训练模型并应用量化
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,  # 模型标识符
    quantization_config=quantization_config,  # 应用量化配置
    device_map="auto",  # 自动分配设备（GPU/CPU）
    trust_remote_code=True,  # 信任远程代码（某些模型需要）
    attn_implementation="eager"  # 使用eager注意力实现，修复警告
)
# 禁用缓存以节省内存（训练时不需要缓存）
model.config.use_cache = False

# 加载分词器（将文本转换为模型可理解的数字）
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# 如果没有填充token，使用结束token作为填充token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 设置填充方向为右侧（在序列末尾添加填充）
tokenizer.padding_side = "right"
print("Model and tokenizer loaded.")

# --- 4. PEFT (LoRA) 配置 ---
print("Step 3: Configuring LoRA...")
# LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法
# 它不修改原模型参数，而是添加小的适配器层
peft_config = LoraConfig(
    r=8,  # LoRA的秩，控制适配器的大小。越小训练越快，但表达能力越弱
    lora_alpha=16,  # LoRA的缩放参数，通常设为r的2倍
    lora_dropout=0.1,  # Dropout概率，防止过拟合
    bias="none",  # 不训练偏置参数
    task_type="CAUSAL_LM",  # 任务类型：因果语言模型（生成式模型）
    target_modules=['q_proj', 'v_proj', 'o_proj'],  # 只对这些注意力层应用LoRA
)
print("LoRA configured.")


# --- 5. SFT训练配置 ---
print("Step 4: Setting up SFTConfig...")

# SFTConfig包含所有训练相关的参数设置
training_args = SFTConfig(
    # 基础训练参数
    output_dir=OUTPUT_DIR,  # 模型保存目录
    per_device_train_batch_size=2,  # 每个设备的批次大小，设为2避免显存不足
    gradient_accumulation_steps=4,  # 梯度累积步数，相当于批次大小为2*4=8
    learning_rate=2e-4,  # 学习率，控制参数更新的步长
    num_train_epochs=15,  # 训练轮数，数据集会被遍历15次
    
    # 保存策略
    save_strategy="epoch",  # 每个epoch结束后保存模型
    save_steps=500,  # 每500步保存一次（当save_strategy不是steps时此参数无效）
    
    # 日志记录
    logging_steps=1,  # 每1步记录一次日志
    logging_strategy="steps",  # 按步数记录日志
    logging_first_step=True,  # 记录第一步的日志
    report_to="tensorboard",  # 使用TensorBoard记录训练过程
    log_level="info",  # 日志级别：信息级别
    logging_nan_inf_filter=True,  # 过滤掉NaN和无穷大值的日志
    
    # 验证策略
    eval_strategy="no",  # 不进行验证（因为我们没有验证集）
    
    # 优化器设置
    optim="adamw_torch",  # 使用AdamW优化器
    lr_scheduler_type="cosine",  # 使用余弦退火学习率调度器
    warmup_ratio=0.1,  # 前10%的步数用于学习率预热
    
    # 精度设置
    bf16=True,  # 使用bfloat16混合精度训练，节省显存并加速训练
    
    # 数据加载设置
    dataloader_drop_last=True,  # 丢弃最后一个不完整的批次
    remove_unused_columns=False,  # 保留数据集中的所有列
    
    # 进度显示
    disable_tqdm=False,  # 启用进度条显示
)
print("SFTConfig set.")

# --- 6. 初始化并开始训练 ---
print("Step 5: Initializing SFT Trainer...")

# 创建SFT训练器，整合所有组件
trainer = SFTTrainer(
    model=model,  # 要训练的模型
    args=training_args,  # 训练配置参数
    train_dataset=dataset,  # 训练数据集
    processing_class=tokenizer,  # 分词器，用于处理文本数据
    peft_config=peft_config,  # LoRA配置，用于参数高效微调
)

print("Starting training...")
# 开始训练过程
# 这会根据配置的参数进行多轮训练，自动保存检查点和日志
trainer.train()

print("Training complete. Saving model...")
# 训练完成后保存最终模型
# 这会保存LoRA适配器到输出目录，而不是整个模型
trainer.save_model()
print(f"Model saved to {OUTPUT_DIR}")