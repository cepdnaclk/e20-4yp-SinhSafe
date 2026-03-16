import torch
import pandas as pd
import os
# --- 1. CRITICAL: GPU & MEMORY TARGETING ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import shutil
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
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
TOKENIZER_ID = "polyglots/Extended-Sinhala-LLaMA"

DATA_DIR = os.path.join(os.getcwd(), "data", "processed_ground_truth", "v2")
OUTPUT_DIR = "./results/sinhsafe_sinllama_v2"
FINAL_MODEL_DIR = "./models/sinhsafe_sinllama_run2"

# --- 3. LOAD TOKENIZER & BASE MODEL (PURE BFLOAT16) ---
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
model = PeftModel.from_pretrained(model, ADAPTER_ID, is_trainable=True)

# --- 5. DATA PREPARATION (V2 DATA) ---
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

val_df_raw = None

def format_dataset(df):
    formatted_texts = []
    for _, row in df.iterrows():
        text = alpaca_prompt.format(row['cleaned_text'], row['label']) + tokenizer.eos_token
        formatted_texts.append(text)
    return Dataset.from_pandas(pd.DataFrame({'text': formatted_texts}))

def load_and_split_data():
    global val_df_raw
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
    
    train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    max_size = train_df['label'].value_counts().max()
    train_bal = pd.concat([
        train_df[train_df['label'] == 'Harassment'].sample(max_size, replace=True, random_state=42),
        train_df[train_df['label'] == 'Offensive'].sample(max_size, replace=True, random_state=42),
        train_df[train_df['label'] == 'Normal'].sample(max_size, replace=True, random_state=42)
    ]).sample(frac=1, random_state=42)
    
    test_df.to_csv("sinhsafe_sinllama_v2_test_set.csv", index=False)
    print(f"📁 Saved {len(test_df)} unseen test rows to 'sinhsafe_sinllama_v2_test_set.csv'")

    val_df_raw = val_df.copy()

    return format_dataset(train_bal), format_dataset(val_df)

train_dataset, val_dataset = load_and_split_data()

# --- 6. SMART CHECKPOINTING TRAINING LOOP ---
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, 
    args=SFTConfig(
        dataset_text_field="text",
        max_length=512,
        
        # --- EXTREME MEMORY SAVERS ---
        per_device_train_batch_size=1,    # Absolute minimum memory footprint per step
        per_device_eval_batch_size=1,     # Absolute minimum memory footprint per step
        gradient_accumulation_steps=32,   # 1 * 32 = Effective Batch Size of 32
        gradient_checkpointing=True,      # Discards intermediate activations
        optim="paged_adamw_8bit",         # Offloads optimizer math to save ~3GB VRAM
        
        warmup_steps=100,
        num_train_epochs=2,       
        learning_rate=5e-5,       
        weight_decay=0.05,        
        bf16=True, 
        logging_steps=10,
        
        # --- SMART STORAGE & EARLY STOPPING ---
        eval_strategy="steps",    
        eval_steps=50,            
        save_strategy="steps",    
        save_steps=50,            
        save_total_limit=1,           
        load_best_model_at_end=True,  
        metric_for_best_model="eval_loss",
        greater_is_better=False,      
        
        output_dir=OUTPUT_DIR,
        report_to="none"
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
)

print(f"🚀 Starting SinLLaMA V2 Training in Pure bfloat16 on GPU 0...")
trainer.train()

trainer.model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"✅ Training complete! The absolute best model has been safely saved to {FINAL_MODEL_DIR}")


# =====================================================================
# 7. SUPERVISOR REPORT EXTRACTION
# =====================================================================
print("\n" + "="*60)
print("🔄 Running final evaluation on Validation Set to generate report...")

model.eval()
true_labels = val_df_raw['label'].tolist()
pred_labels = []

inference_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Categorize the following Sinhala/Singlish text into exactly one of three categories: Normal, Offensive, or Harassment. 
Use the following strict definitions to make your decision:
- Harassment: Targeted behavior meant to degrade or intimidate. Includes threats of violence, encouraging self-harm, or attacking someone's family, women, religion, and ethnicity-based attacks.
- Offensive: Content that violates social norms or decorum. General vulgarity, crude jokes, or "blue" humor without a specific target.
- Normal: Standard, respectful communication. Professional, casual, or friendly dialogue that follows social etiquette.

### Input:
{}

### Response:
"""

for text in tqdm(val_df_raw['cleaned_text'], desc="Evaluating"):
    prompt = inference_prompt.format(text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("### Response:\n")[-1].strip()
    
    if "Harassment" in answer:
        pred_labels.append("Harassment")
    elif "Offensive" in answer:
        pred_labels.append("Offensive")
    else:
        pred_labels.append("Normal")

acc = accuracy_score(true_labels, pred_labels)
prec, rec, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)

report = "\n" + "="*60 + "\n"
report += "📊 1. DATA FOR LOSS CURVES (OVERFITTING PROOF)\n"
report += "="*60 + "\n"

history = trainer.state.log_history
extracted_logs = {}

for log in history:
    step = log.get('step')
    if step not in extracted_logs:
        extracted_logs[step] = {'train_loss': 'N/A', 'eval_loss': 'N/A', 'epoch': log.get('epoch', 0)}
    if 'loss' in log:
        extracted_logs[step]['train_loss'] = round(log['loss'], 4)
    if 'eval_loss' in log:
        extracted_logs[step]['eval_loss'] = round(log['eval_loss'], 4)

report += f"{'Step':<10} | {'Epoch':<10} | {'Train Loss':<15} | {'Eval Loss':<15}\n"
report += "-" * 55 + "\n"
for step in sorted(extracted_logs.keys()):
    data = extracted_logs[step]
    if data['train_loss'] != 'N/A' or data['eval_loss'] != 'N/A':
        ep = f"{data['epoch']:.2f}"
        report += f"{step:<10} | {ep:<10} | {data['train_loss']:<15} | {data['eval_loss']:<15}\n"

report += "\n" + "="*60 + "\n"
report += "📈 2. DATA FOR PERFORMANCE METRICS (BEST MODEL)\n"
report += "="*60 + "\n"
report += f"Accuracy     : {acc:.4f}\n"
report += f"Precision (W): {prec:.4f}\n"
report += f"Recall (W)   : {rec:.4f}\n"
report += f"F1-Score (W) : {f1:.4f}\n"

report += "\n" + "="*60 + "\n"
report += "🏆 3. SELECTION LOGIC\n"
report += "="*60 + "\n"
report += "Selection Method: Checkpoint-based Early Stopping with lowest Validation Loss.\n"
report += f"Winning Model Path: {os.path.abspath(FINAL_MODEL_DIR)}\n"
report += "="*60 + "\n"

print(report)

report_path = "sinllama_final_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
    
print(f"✅ Clean report successfully saved to {os.path.abspath(report_path)}")