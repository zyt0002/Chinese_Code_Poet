import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import os

# --- Configuration ---
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DATASET_FILE = "dataset.jsonl"
OUTPUT_DIR = "./code_poet_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Dataset ---
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

# Format the dataset for SFT training
def format_prompt(example):
    return {"text": f"用户: {example['prompt']}\n助手: {example['response']}"}

dataset = dataset.map(format_prompt)
print(f"Dataset loaded with {len(dataset)} examples")
print("Sample:", dataset[0]["text"][:100])

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer...")

# Use BitsAndBytesConfig for proper 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"  # Fix attention warning
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Model and tokenizer loaded successfully!")

# --- LoRA Configuration ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules='all-linear',
)

# --- Training Configuration ---
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # Reduced for testing
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=1,  # Just 1 epoch for testing
    save_strategy="epoch",
    logging_steps=5,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    bf16=True,
    max_length=512,  # Reduced for testing
    packing=False,
    dataset_text_field="text",
)

# --- Initialize Trainer ---
print("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset.select(range(10)),  # Use only first 10 examples for testing
    processing_class=tokenizer,
    peft_config=peft_config,
)

print("Configuration successful! Ready to train.")
print("To start training, uncomment the next lines:")
print("# trainer.train()")
print("# trainer.save_model()")