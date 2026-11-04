import os
import subprocess
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./output/merged_model"
GGUF_PATH = os.path.expanduser("~/mindsync-0.5b.gguf")

print("🚀 Step 1: Loading Base Model & Tokenizer...")
# 1. Khởi tạo tokenizer TRƯỚC
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float32, device_map="cpu"
)

print("⚡ Step 2: Applying LoRA Config...")
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

print("📚 Step 3: Loading & Processing Dataset...")
# 2. Định nghĩa hàm tokenize (lúc này biến tokenizer đã tồn tại)
def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=512,
        padding=False
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

# 3. Load dataset và thực hiện map
dataset = load_dataset("json", data_files={"train": "dataset.jsonl"})
tokenized_ds = dataset["train"].map(tokenize, remove_columns=["text"])

print("🏋️ Step 4: Training with LoRA...")
training_args = TrainingArguments(
    output_dir="./output/lora_checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-4,
    logging_steps=1,
    use_cpu=True,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
)
trainer.train()

print("🔗 Step 5: Merging LoRA Weights into Base Model...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("📦 Step 6: Converting to GGUF format for Ollama...")
if not os.path.exists("/tmp/llama.cpp"):
    subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"], check=True)

subprocess.run([
    "python3", "/tmp/llama.cpp/convert_hf_to_gguf.py",
    OUTPUT_DIR, "--outfile", GGUF_PATH, "--outtype", "f16"
], check=True)

print(f"✨ Model successfully converted and saved at: {GGUF_PATH}")