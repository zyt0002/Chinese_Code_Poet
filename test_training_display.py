#!/usr/bin/env python3

# 测试训练显示配置
from trl import SFTConfig
from transformers import TrainerCallback

print("=== 测试训练显示配置 ===")

# 测试SFTConfig参数
training_args = SFTConfig(
    output_dir="./test_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=15,
    save_strategy="epoch",
    save_steps=500,
    logging_steps=1,
    logging_strategy="steps",
    eval_strategy="no",
    report_to="tensorboard",
    logging_first_step=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    dataloader_drop_last=True,
    remove_unused_columns=False,
    disable_tqdm=False,
    log_level="info",
    logging_nan_inf_filter=True,
)

print("✅ SFTConfig 创建成功")
print(f"✅ 日志步数: {training_args.logging_steps}")
print(f"✅ 进度条启用: {not training_args.disable_tqdm}")
print(f"✅ 日志级别: {training_args.log_level}")

# 测试回调函数
class ProgressCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n🚀 开始 Epoch {int(state.epoch) + 1}/{args.num_train_epochs}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"✅ 完成 Epoch {int(state.epoch)}/{args.num_train_epochs}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            current_step = state.global_step
            total_steps = state.max_steps if state.max_steps > 0 else "Unknown"
            current_epoch = state.epoch
            print(f"Step {current_step}/{total_steps} | Epoch {current_epoch:.2f} | Loss: {logs['loss']:.4f}")

callback = ProgressCallback()
print("✅ 进度回调函数创建成功")

# 模拟训练信息计算
dataset_size = 285
batch_size = training_args.per_device_train_batch_size
grad_accum = training_args.gradient_accumulation_steps
epochs = training_args.num_train_epochs

effective_batch_size = batch_size * grad_accum
steps_per_epoch = dataset_size // effective_batch_size
total_steps = steps_per_epoch * epochs

print(f"\n=== 训练配置信息 ===")
print(f"数据集大小: {dataset_size}")
print(f"批次大小: {batch_size}")
print(f"梯度累积步数: {grad_accum}")
print(f"有效批次大小: {effective_batch_size}")
print(f"每轮步数: {steps_per_epoch}")
print(f"总轮数: {epochs}")
print(f"预计总步数: {total_steps}")

print("\n✅ 训练显示配置测试完成！现在应该能看到详细的epoch和步数信息。")