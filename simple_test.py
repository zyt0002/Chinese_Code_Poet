#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys

def main():
    print("=== 代码诗人测试 ===")
    
    # 配置
    BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
    ADAPTER_PATH = "./code_poet_final_guaranteed"
    
    try:
        print("1. 加载基础模型...")
        
        # 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        print("2. 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("3. 加载LoRA适配器...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()
        
        print("✅ 模型加载成功！")
        
        # 测试函数 - 优化参数确保完整回答
        def ask_poet(question, max_tokens=256):
            prompt = f"用户: {question}\n助手:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,  # 增加生成长度
                    min_new_tokens=20,  # 确保最少生成20个token
                    temperature=0.7,  # 稍微降低温度获得更稳定输出
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=0.85,  # 稍微降低top_p
                    top_k=50,  # 添加top_k限制
                    repetition_penalty=1.15,  # 稍微增加重复惩罚
                    length_penalty=1.0,  # 添加长度惩罚
                    no_repeat_ngram_size=3,  # 避免3-gram重复
                    use_cache=False,  # 禁用缓存以避免DynamicCache错误
                    early_stopping=False  # 不提前停止
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "助手:" in response:
                answer = response.split("助手:")[-1].strip()
                # 清理可能的截断标记
                if answer.endswith("..."):
                    answer = answer[:-3].strip()
                return answer
            return response
        
        # 测试
        print("\n=== 开始测试 ===")
        test_questions = [
            "解释一下Python的for循环",
            "什么是递归？",
            "解释一下面向对象编程",
            "什么是装饰器？",
            "解释一下异常处理"
        ]
        
        for i, q in enumerate(test_questions, 1):
            print(f"\n{i}. 问题: {q}")
            answer = ask_poet(q)
            print(f"   回答: {answer}")
        
        print("\n=== 交互模式 ===")
        print("输入问题测试模型，输入'quit'退出")
        
        while True:
            try:
                question = input("\n你的问题: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if question:
                    answer = ask_poet(question)
                    print(f"代码诗人: {answer}")
            except KeyboardInterrupt:
                break
        
        print("\n再见！")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()