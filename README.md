# 中文代码诗人 🎭

一个使用LoRA技术微调Phi-3模型的中文诗意编程助手，能够以优雅的诗意语言解释编程概念。

## 📋 项目简介

本项目基于微软的Phi-3-mini-4k-instruct模型，通过参数高效微调(PEFT)技术，训练出一个能够用诗意语言解释编程概念的AI助手。模型使用LoRA(低秩适应)技术进行微调，既保持了原模型的能力，又赋予了其独特的诗意表达风格。

### ✨ 特色功能

- 🎨 **诗意编程解释**: 用优雅的中文诗意语言解释复杂的编程概念
- 🚀 **高效训练**: 使用LoRA技术，只需训练少量参数
- 💾 **内存优化**: 4位量化技术，降低显存需求
- 🔧 **易于使用**: 简单的训练和推理脚本

## 🛠️ 环境要求

### 硬件要求
- GPU: 建议8GB+显存的NVIDIA GPU
- RAM: 16GB以上系统内存
- 存储: 至少10GB可用空间

### 软件要求
- Python 3.8+
- CUDA 11.8+ (如果使用GPU)

## 📦 安装依赖

```bash
# 安装完整依赖
pip install -r requirements.txt

# 或安装最小依赖
pip install -r requirements-minimal.txt
```

### 主要依赖包
- `torch>=2.0.0`: PyTorch深度学习框架
- `transformers>=4.36.0`: Hugging Face模型库
- `peft>=0.7.0`: 参数高效微调库
- `trl>=0.7.0`: 强化学习库
- `bitsandbytes>=0.41.0`: 量化库

## 🎯 快速开始

### 1. 训练模型

```bash
python train.py
```

训练脚本会：
- 加载预训练的Phi-3模型
- 应用LoRA配置进行参数高效微调
- 使用诗意编程问答数据集进行训练
- 保存训练好的适配器到`./code_poet_final_guaranteed`

### 2. 测试模型

```bash
python simple_test.py
```

测试脚本提供：
- 预设问题的自动测试
- 交互式问答模式
- 生成质量评估

### 3. 数据验证

```bash
# 验证数据格式
python validate_json.py

# 测试数据格式
python test_data_format.py
```

## 📊 数据集格式

项目使用JSON Lines格式的训练数据，每行包含一个问答对：

```json
{"prompt": "解释一下Python的for循环", "response": "在元素的序列中漫步，一次取一个，直到旅程的尽头，这便是for的优雅巡游。"}
```

### 数据集文件
- `dataset.jsonl`: 原始训练数据集
- `ai_studio_code.jsonl`: 处理后的训练数据

## 🏗️ 项目结构

```
1_chinese_poet/
├── train.py                    # 主训练脚本
├── simple_test.py             # 模型测试脚本
├── validate_json.py           # 数据验证脚本
├── test_data_format.py        # 数据格式测试
├── requirements.txt           # 完整依赖列表
├── requirements-minimal.txt   # 最小依赖列表
├── dataset.jsonl             # 原始训练数据
├── ai_studio_code.jsonl      # 处理后训练数据
├── training_config_summary.md # 训练配置总结
├── code_poet_final_guaranteed/ # 训练输出目录
│   ├── adapter_config.json    # LoRA配置文件
│   ├── adapter_model.safetensors # LoRA权重文件
│   └── checkpoint-*/          # 训练检查点
└── code_poet_test/           # 测试输出目录
```

## ⚙️ 配置参数

### 训练配置
- **基础模型**: `microsoft/Phi-3-mini-4k-instruct`
- **LoRA秩**: 8
- **学习率**: 2e-4
- **训练轮数**: 15
- **批次大小**: 2 (梯度累积4步)
- **序列长度**: 2048

### 生成配置
- **最大新token数**: 256
- **最小新token数**: 20
- **温度**: 0.7
- **Top-p**: 0.85
- **重复惩罚**: 1.15

## 🎨 示例输出

**问题**: "什么是递归？"

**回答**: "函数在自身的深处呼唤自己，如同一面映照着另一面的镜子，直至最初的条件被满足，幻境才逐层消散。"

**问题**: "解释一下Git"

**回答**: "Git是时间的魔法师，它记录下代码的每一次呼吸与心跳。'Commit'是封存此刻的咒语，'Branch'是通往平行世界的岔路，而'Merge'，则是让所有溪流汇入江河的仪式。"

## 🔧 自定义训练

### 修改训练数据
1. 编辑`ai_studio_code.jsonl`文件
2. 确保每行都是有效的JSON格式
3. 包含`prompt`和`response`字段

### 调整训练参数
在`train.py`中修改以下参数：
- `num_train_epochs`: 训练轮数
- `learning_rate`: 学习率
- `per_device_train_batch_size`: 批次大小
- LoRA配置参数

### 修改生成参数
在`simple_test.py`中调整生成参数：
- `temperature`: 控制随机性
- `max_new_tokens`: 最大生成长度
- `top_p`: 核采样参数

## 📈 监控训练

训练过程会生成TensorBoard日志：

```bash
tensorboard --logdir=./code_poet_final_guaranteed/runs
```

## ⚠️ 注意事项

1. **显存需求**: 即使使用4位量化，仍需要足够的GPU显存
2. **训练时间**: 完整训练可能需要几小时到几天，取决于硬件配置
3. **数据质量**: 训练效果很大程度上取决于训练数据的质量
4. **过拟合**: 小数据集容易过拟合，注意调整训练轮数

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目遵循MIT许可证。

## 🙏 致谢

- Microsoft Phi-3团队提供的优秀基础模型
- Hugging Face团队的transformers库
- PEFT库的开发者们
- 所有为开源AI做出贡献的开发者们