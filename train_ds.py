"""
Phi-3-mini LoRA 微调演示脚本
使用自定义数据集对模型进行指令微调
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# 常量定义
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"  # 基础模型
DATASET_PATH = "dataset.jsonl"  # 数据集路径
OUTPUT_DIR = "code_poet_model_final"  # 输出目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测设备

# 数据集格式化函数 - 使用 Phi-3 的指令模板
def format_instruction(sample):
    return f"<|user|>\n{sample['prompt']}<|end|>\n<|assistant|>\n{sample['response']}<|end|>\n"

def main():
    # 1. 加载并准备数据集
    print("加载数据集...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # 划分训练集和验证集 (90% 训练, 10% 验证)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # 格式化数据集
    def format_dataset(samples):
        # 修复: 正确处理批处理数据
        formatted_texts = []
        for i in range(len(samples["prompt"])):
            sample = {
                "prompt": samples["prompt"][i],
                "response": samples["response"][i]
            }
            formatted_texts.append(format_instruction(sample))
        return {"text": formatted_texts}
    
    # 应用格式化
    dataset = dataset.map(
        format_dataset,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 2. 加载模型和分词器
    print("加载模型和分词器...")
    
    # 配置4位量化以节省内存
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    
    # 3. 配置LoRA
    peft_config = LoraConfig(
        r=16,  # LoRA秩
        lora_alpha=32,  # LoRA alpha参数
        lora_dropout=0.05,  # Dropout概率
        bias="none",  # 偏置类型
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. 配置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,  # 根据GPU内存调整
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,  # 使用混合精度训练
        report_to="none",  # 禁用默认的报告器
    )
    
    # 5. 创建SFT训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=2048,  # 根据模型和内存调整
        tokenizer=tokenizer,
        packing=False,  # 不打包序列以提高效率
    )
    
    # 6. 开始训练
    print("开始训练...")
    trainer.train()
    
    # 7. 保存最终模型
    print("训练完成，保存模型...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"模型已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()