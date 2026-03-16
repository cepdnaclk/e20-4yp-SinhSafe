import torch
import pandas as pd
import os
# --- 1. CRITICAL: GPU & MEMORY TARGETING ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

# --- 2. CONFIGURATION ---
BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B"  
ADAPTER_ID = "./SinLlama_Local"                 
TOKENIZER_ID = "polyglots/Extended-Sinhala-LLaMA"

DATA_DIR = os.path.join(os.getcwd(), "data", "processed_ground_truth", "v2")
SCRATCH_PATH = "/scratch1/e20-4yp-sinhsafe"
OUTPUT_DIR = os.path.join(SCRATCH_PATH, "results", "sinhsafe_sinllama_prod_temp")
FINAL_MODEL_DIR = os.path.join(SCRATCH_PATH, "models", "sinllama_production")
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

EPOCHS = 1

# --- 3. LOAD TOKENIZER & BASE MODEL (PURE BFLOAT16) ---
print("📂 Loading Tokenizer and Base Model...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
tokenizer.pad_token = tokenizer.eos_token

# Pure bfloat16 - No BitsAndBytes compression!
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True 
)

model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

# --- 4. LOAD THE SINLLAMA ADAPTER ---
print("🔗 Attaching PEFT Adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_ID, is_trainable=True)

# --- 5. DATA PREPARATION (100% DATA) ---
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

def load_data():
    print("📊 Loading 100% of V2 data for Production Training...")
    path_harass = os.path.join(DATA_DIR, "processed_final_harassment.csv")
    path_offen  = os.path.join(DATA_DIR, "processed_final_offensive.csv")
    path_norm   = os.path.join(DATA_DIR, "processed_final_normal.csv")

    df_harass = pd.read_csv(path_harass)
    df_offen  = pd.read_csv(path_offen)
    df_norm   = pd.read_csv(path_norm)
    
    df_norm['label'] = "Normal"
    df_offen['label'] = "Offensive"
    df_harass['label'] = "Harassment"
    
    df_all = pd.concat([df_harass, df_offen, df_norm]).dropna(subset=['cleaned_text', 'label'])
    
    # Balance 100% of the dataset
    max_size = df_all['label'].value_counts().max()
    train_bal = pd.concat([
        df_all[df_all['label'] == 'Harassment'].sample(max_size, replace=True, random_state=42),
        df_all[df_all['label'] == 'Offensive'].sample(max_size, replace=True, random_state=42),
        df_all[df_all['label'] == 'Normal'].sample(max_size, replace=True, random_state=42)
    ]).sample(frac=1, random_state=42)
    
    print(f"✅ Loaded {len(train_bal)} balanced rows.")
    return format_dataset(train_bal)

train_dataset = load_data()

# --- 6. PRODUCTION TRAINING LOOP ---
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=512,
        
        # --- EXTREME MEMORY SAVERS ---
        per_device_train_batch_size=1,    # Minimum memory footprint per step
        gradient_accumulation_steps=32,   # Effective Batch Size of 32
        gradient_checkpointing=True,      
        optim="paged_adamw_8bit",         
        
        warmup_steps=100,
        num_train_epochs=EPOCHS,          # Locked to 1 Epoch
        learning_rate=5e-5,       
        weight_decay=0.05,        
        bf16=True, 
        logging_steps=10,
        
        # --- PURE TRAINING (No Eval) ---
        save_strategy="no",               # Save only at the very end manually
        output_dir=OUTPUT_DIR,
        report_to="none"
    ),
)

print(f"\n🚀 Starting PRODUCTION SinLLaMA Training on GPU 1...")
trainer.train()

# --- 7. SAVE PRODUCTION MODEL ---
print(f"\n💾 Saving Final Production Model to Scratch Drive...")
trainer.model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"🌟 Success! SinLLaMA Production Model is safely stored at:\n{os.path.abspath(FINAL_MODEL_DIR)}")