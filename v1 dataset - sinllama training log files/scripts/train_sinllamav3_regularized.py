import os
# --- 1. CRITICAL: GPU & CACHE TARGETING ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import PeftModel
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# --- 2. CONFIGURATION ---
BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B"  
ADAPTER_ID = "./SinLlama_Local"                 
TOKENIZER_ID = "./SinLlama_Local" 
OUTPUT_DIR = "./results/sinhsafe_llama_v3_temp"

# --- 3. LOAD TOKENIZER & BASE MODEL ---
print("📂 Loading weights from local cache...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
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
    low_cpu_mem_usage=True,
    local_files_only=True
)

# --- CRITICAL: FIX VOCAB SIZE MISMATCH ---
print(f"📏 Resizing embeddings to match tokenizer ({len(tokenizer)})...")
model.resize_token_embeddings(len(tokenizer))

print("🔗 Attaching PEFT Adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_ID, is_trainable=True)

# --- 4. DATA PREPARATION ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Categorize the following Sinhala/Singlish text into exactly one of three categories: Normal, Offensive, or Harassment. 
Use the following strict definitions:
- Harassment: Targeted behavior meant to degrade or intimidate. Includes threats of violence, encouraging self-harm, or attacking someone's family, women, religion, and ethnicity-based attacks.
- Offensive: Content that violates social norms or decorum. General vulgarity, crude jokes, or "blue" humor without a specific target.
- Normal: Standard, respectful communication. Professional, casual, or friendly dialogue that follows social etiquette.

### Input:
{}

### Response:
{}"""

val_df_raw = None

def format_dataset(df):
    formatted_texts = [alpaca_prompt.format(row['cleaned_text'], row['label']) + tokenizer.eos_token for _, row in df.iterrows()]
    return Dataset.from_pandas(pd.DataFrame({'text': formatted_texts}))

def load_and_split_data():
    global val_df_raw
    print("📊 Loading and splitting Excel datasets...")
    df_harass = pd.read_excel("data/processed_ground_truth/processed_consolidated_harassment.xlsx")
    df_offen = pd.read_excel("data/processed_ground_truth/processed_consolidated_offensive.xlsx")
    df_norm = pd.read_excel("data/processed_ground_truth/processed_consolidated_normal.xlsx")
    
    df_norm['label'], df_offen['label'], df_harass['label'] = "Normal", "Offensive", "Harassment"
    df_all = pd.concat([df_harass, df_offen, df_norm]).dropna(subset=['cleaned_text'])
    
    train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['label'])
    val_df, _ = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    val_df_raw = val_df.copy()
    
    max_size = train_df['label'].value_counts().max()
    train_bal = pd.concat([train_df[train_df['label'] == lbl].sample(max_size, replace=True, random_state=42) for lbl in ["Normal", "Offensive", "Harassment"]]).sample(frac=1, random_state=42)
    
    return format_dataset(train_bal), format_dataset(val_df)

train_dataset, val_dataset = load_and_split_data()

# --- 5. TRAINING (REGULARIZED MODE) ---
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, 
    args=SFTConfig(
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=5e-5,
        weight_decay=0.05,
        bf16=True,
        logging_steps=10,
        
        # --- EARLY STOPPING COMPLIANCE SETTINGS ---
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",            # Must be 'steps' for EarlyStopping
        save_steps=50,                    # Match eval_steps
        load_best_model_at_end=True,      # Required for EarlyStopping logic
        metric_for_best_model="eval_loss", # Crucial fix for the AssertionError
        save_total_limit=1,               # Overwrites old save to keep disk usage low
        
        report_to="none",
        output_dir=OUTPUT_DIR
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("🚀 Starting SinLLaMA V3 Training on GPU 1...")
trainer.train()

# --- 6. SUPERVISOR REPORT GENERATION ---
print("\n" + "="*60 + "\n🔄 GENERATING FINAL SUPERVISOR REPORT\n" + "="*60)

history = trainer.state.log_history
report_content = f"{'Step':<10} | {'Epoch':<10} | {'Train Loss':<15} | {'Eval Loss':<15}\n"
report_content += "-" * 55 + "\n"
extracted_logs = {}
for log in history:
    step = log.get('step')
    if step not in extracted_logs: extracted_logs[step] = {'t': 'N/A', 'e': 'N/A', 'ep': log.get('epoch', 0)}
    if 'loss' in log: extracted_logs[step]['t'] = round(log['loss'], 4)
    if 'eval_loss' in log: extracted_logs[step]['e'] = round(log['eval_loss'], 4)

for s in sorted(extracted_logs.keys()):
    if extracted_logs[s]['t'] != 'N/A' or extracted_logs[s]['e'] != 'N/A':
        report_content += f"{s:<10} | {extracted_logs[s]['ep']:<10.2f} | {extracted_logs[s]['t']:<15} | {extracted_logs[s]['e']:<15}\n"

# B. Classification Inference Pass
model.eval()
true_labels = val_df_raw['label'].tolist()
pred_labels = []

inf_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Categorize the following Sinhala/Singlish text into exactly one of three categories: Normal, Offensive, or Harassment. 

### Input:
{}

### Response:
"""

print("\n🔍 Running Inference Pass for Metrics...")
for text in tqdm(val_df_raw['cleaned_text']):
    inputs = tokenizer(inf_prompt.format(text), return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[-1].strip()
    if "Harassment" in answer: pred_labels.append("Harassment")
    elif "Offensive" in answer: pred_labels.append("Offensive")
    else: pred_labels.append("Normal")

acc = accuracy_score(true_labels, pred_labels)
prec, rec, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)

final_block = "\n" + "="*60 + "\n📈 PERFORMANCE SUMMARY (BEST EPOCH)\n" + "="*60 + "\n"
final_block += f"Accuracy     : {acc:.4f}\nPrecision (W): {prec:.4f}\nRecall (W)   : {rec:.4f}\nF1-Score (W) : {f1:.4f}\n"
final_block += "="*60 + "\n"

print(report_content + final_block)
with open("final_llama_v3_report.txt", "w", encoding="utf-8") as f:
    f.write(report_content + final_block)
print("✅ Supervisor Report saved to final_llama_v3_report.txt")