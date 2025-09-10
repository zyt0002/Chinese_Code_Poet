import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- 配置 ---
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "./code_poet_final_guaranteed"

print("=== 加载训练好的代码诗人模型 ===")

# --- 加载基础模型和分词器 ---
print("1. 加载基础模型和分词器...")

# 使用相同的量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("2. 加载LoRA适配器...")
# 加载训练好的LoRA适配器
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("模型加载完成！")

# --- 测试函数 ---
def generate_response(prompt, max_length=300, temperature=0.7):
    """生成诗意的代码解释"""
    # 格式化输入
    formatted_prompt = f"用户: {prompt}\n助手:"
    
    # 编码输入
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            min_new_tokens=20,
            temperature=temperature,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取助手的回答
    if "助手:" in response:
        response = response.split("助手:")[-1].strip()
    
    return response

# --- 交互测试 ---
print("\n=== 开始测试代码诗人 ===")
print("输入编程问题，模型会给出诗意的解释。输入 'quit' 退出。\n")

# 预设测试问题
test_questions = [
    "什么是Python的列表推导式？",
    "解释一下函数的概念",
    "什么是面向对象编程？",
    "解释一下异常处理",
    "什么是装饰器？"
]

print("=== 预设问题测试 ===")
for i, question in enumerate(test_questions, 1):
    print(f"\n{i}. 问题: {question}")
    response = generate_response(question)
    print(f"   答案: {response}")
    print("-" * 60)

print("\n=== 交互模式 ===")
while True:
    user_input = input("\n请输入你的编程问题: ").strip()
    
    if user_input.lower() in ['quit', 'exit', '退出', 'q']:
        print("再见！")
        break
    
    if not user_input:
        continue
    
    print("思考中...")
    response = generate_response(user_input)
    print(f"代码诗人: {response}")