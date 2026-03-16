import os
# --- GPU TARGETING ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Using GPU 1 or 2
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B"  
ADAPTER_ID = "./SinLlama_Local"                 
TOKENIZER_ID = "polyglots/Extended-Sinhala-LLaMA"

# --- MODEL PATH (Strictly Scratch Drive) ---
SCRATCH_PATH = "/scratch1/e20-4yp-sinhsafe"
OUTPUT_DIR = os.path.join(SCRATCH_PATH, "temp_sinllama_results")
FINAL_MODEL_DIR = os.path.join(SCRATCH_PATH, "sinllama_prod_model")

# ==========================================
# 2. LOAD TOKENIZER & BASE MODEL
# ==========================================
print("⚙️ Loading Tokenizer and Base LLaMA 3 Model in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Native BF16 for Ada
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True 
)

model.resize_token_embeddings(len(tokenizer))

# --- LOAD THE SINLLAMA ADAPTER ---
print("🔌 Attaching the SinLlama Adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_ID, is_trainable=True)

# ==========================================
# 3. DATA PREPARATION (100% Data + Balanced)
# ==========================================
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Categorize the following Sinhala/Singlish text into exactly one of three categories: Normal, Offensive, or Harassment. 
Use the following strict definitions to make your decision:
- Harassment: Targeted behavior meant to degrade or intimidate. Includes threats of violence, encouraging self-harm, or attacking someone's family, women, religion, and ethnicity-based attacks.
- Offensive: Content that violates social norms or decorum. General vulgarity, crude jokes, or "blue" humor without a specific target.
- Normal: Standard, respectful communication. Professional, casual, or friendly dialogue that follows social etiquette.

### Input:
{}

### Response:
{}"""

def format_dataset(df):
    formatted_texts = []
    for _, row in df.iterrows():
        text = alpaca_prompt.format(row['cleaned_text'], row['label']) + tokenizer.eos_token
        formatted_texts.append(text)
    return Dataset.from_pandas(pd.DataFrame({'text': formatted_texts}))

def load_and_balance_all_data():
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data", "processed_ground_truth")
    
    df_harass = pd.read_excel(os.path.join(data_dir, "processed_consolidated_harassment.xlsx"))
    df_offen = pd.read_excel(os.path.join(data_dir, "processed_consolidated_offensive.xlsx"))
    df_norm = pd.read_excel(os.path.join(data_dir, "processed_consolidated_normal.xlsx"))
    
    # Apply String Labels
    df_norm['label'] = "Normal"
    df_offen['label'] = "Offensive"
    df_harass['label'] = "Harassment"
    
    df_all = pd.concat([df_harass, df_offen, df_norm]).dropna(subset=['cleaned_text', 'label'])
    print(f"📊 Total raw rows loaded: {len(df_all)}")
    
    # --- BALANCE THE 100% DATASET ---
    max_size = df_all['label'].value_counts().max()
    print(f"⚖️ Oversampling all classes to match the majority class size: {max_size}")
    
    df_bal = pd.concat([
        df_all[df_all['label'] == 'Harassment'].sample(max_size, replace=True, random_state=42),
        df_all[df_all['label'] == 'Offensive'].sample(max_size, replace=True, random_state=42),
        df_all[df_all['label'] == 'Normal'].sample(max_size, replace=True, random_state=42)
    ]).sample(frac=1, random_state=42)
    
    print(f"📈 Final balanced training set size: {len(df_bal)} rows.")
    return format_dataset(df_bal)

train_dataset = load_and_balance_all_data()

# ==========================================
# 4. TRAINING LOOP (1 Epoch, No Eval)
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=1,       # LOCKED AT 1 EPOCH
        learning_rate=5e-5,       
        weight_decay=0.05,        
        bf16=True,                # NATIVE BF16 FOR ADA
        logging_steps=10,
        eval_strategy="no",       # Sprint straight to the finish
        save_strategy="no",       
        output_dir=OUTPUT_DIR,
        report_to="none"
    )
)

print(f"\n🚀 Starting 100% Data Production Training for SinLlama...")
trainer.train()

print(f"\n🌟 Training Complete! Saving SinLlama God Model to {FINAL_MODEL_DIR}...")
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
trainer.model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print(f"🎉 Production Model successfully built and secured in {SCRATCH_PATH}!")