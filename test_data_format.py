#!/usr/bin/env python3

# 测试数据格式是否正确
from datasets import load_dataset

print("=== 测试数据格式修复 ===")

DATASET_FILE = "ai_studio_code.jsonl"
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

def format_prompt(example):
    # SFTTrainer期望prompt和completion字段
    formatted_prompt = f"用户: {example['prompt']}\n助手:"
    completion = f" {example['response']}"
    return {
        "prompt": formatted_prompt,
        "completion": completion
    }

dataset = dataset.map(format_prompt)

print(f"✅ 数据集加载成功: {len(dataset)} 个样本")
print(f"✅ 数据集字段: {dataset.column_names}")
print(f"✅ 包含必需字段: {'prompt' in dataset.column_names and 'completion' in dataset.column_names}")

print("\n=== 示例数据 ===")
example = dataset[0]
print(f"Prompt: {repr(example['prompt'])}")
print(f"Completion: {repr(example['completion'])}")

print("\n=== 测试SFTTrainer期望的格式 ===")
# 模拟SFTTrainer内部的处理
try:
    text = example["prompt"] + example["completion"]
    print(f"✅ 拼接成功: {repr(text[:100])}...")
    print("✅ 数据格式修复成功！SFTTrainer应该能正常工作。")
except KeyError as e:
    print(f"❌ 仍有错误: {e}")