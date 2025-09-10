#!/usr/bin/env python3
import json

def validate_jsonl(file_path):
    """验证JSONL文件格式"""
    errors = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            try:
                data = json.loads(line)
                # 检查必需字段
                if 'prompt' not in data or 'response' not in data:
                    errors.append(f"Line {line_num}: Missing required fields ('prompt' or 'response')")
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON decode error - {e}")
    
    return errors

if __name__ == "__main__":
    file_path = "ai_studio_code.jsonl"
    print(f"Validating {file_path}...")
    
    errors = validate_jsonl(file_path)
    
    if errors:
        print(f"Found {len(errors)} errors:")
        for error in errors:
            print(f"  {error}")
    else:
        print("✅ JSON file is valid!")