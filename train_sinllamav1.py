import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import PeftModel
from trl import SFTTrainer, SFTConfig  # <-- Notice SFTConfig is imported here now

# --- 1. CONFIGURATION ---
BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B"  
ADAPTER_ID = "./SinLlama_Local"                 
TOKENIZER_ID = "polyglots/Extended-Sinhala-LLaMA"
OUTPUT_DIR = "./results/sinhsafe_model"
FINAL_MODEL_DIR = "./models/sinhsafe_final"

# --- 2. LOAD TOKENIZER & BASE MODEL ---
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True 
)

# Resize the base model's brain to match the 139,336 Sinhala tokens
model.resize_token_embeddings(len(tokenizer))

# --- 3. LOAD THE SINLLAMA ADAPTER ---
model = PeftModel.from_pretrained(model, ADAPTER_ID, is_trainable=True)

# --- 4. DATA PREPARATION ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Categorize the following Sinhala/Singlish text into exactly one of three categories: Normal, Offensive, or Harassment.

### Input:
{}

### Response:
{}"""

def load_data():
    df_harass = pd.read_excel("data/processed_ground_truth/processed_consolidated_harassment.xlsx")
    df_offen = pd.read_excel("data/processed_ground_truth/processed_consolidated_offensive.xlsx")
    df_norm = pd.read_excel("data/processed_ground_truth/processed_consolidated_normal.xlsx")
    
    df_norm['label'] = "Normal"
    df_offen['label'] = "Offensive"
    df_harass['label'] = "Harassment"
    
    # --- HANDLE CLASS IMBALANCE ---
    max_size = max(len(df_harass), len(df_offen), len(df_norm))
    
    df_harass_bal = df_harass.sample(max_size, replace=True, random_state=42)
    df_offen_bal = df_offen.sample(max_size, replace=True, random_state=42)
    df_norm_bal = df_norm.sample(max_size, replace=True, random_state=42)
    
    df = pd.concat([df_harass_bal, df_offen_bal, df_norm_bal]).sample(frac=1, random_state=42)
    
    formatted_texts = []
    for _, row in df.iterrows():
        text = alpaca_prompt.format(row['cleaned_text'], row['label']) + tokenizer.eos_token
        formatted_texts.append(text)
        
    return Dataset.from_pandas(pd.DataFrame({'text': formatted_texts}))

dataset = load_data()

# --- 5. TRAINING LOOP ---
# Using the new SFTConfig required by the latest huggingface update
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True, # <--- The fix: Native BFloat16 training
        logging_steps=10,
        output_dir=OUTPUT_DIR,
        report_to="none"
    ),
)

print("🚀 Starting SinhSafe Training on RTX 3090 Ti...")
trainer.train()

trainer.model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"✅ Training complete! Model saved to {FINAL_MODEL_DIR}")