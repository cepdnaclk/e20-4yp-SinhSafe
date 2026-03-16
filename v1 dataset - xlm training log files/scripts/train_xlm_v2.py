import pandas as pd
import numpy as np
import torch
import gc
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.utils import resample
import os

# --- 1. CRITICAL: GPU TARGETING ---
# Ensure you launch with CUDA_VISIBLE_DEVICES=2 in the terminal
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
MODEL_NAME = "xlm-roberta-large"
DATA_DIR = r"data/processed_ground_truth"
N_FOLDS = 5 

id2label = {0: "Normal", 1: "Offensive", 2: "Harassment"}
label2id = {"Normal": 0, "Offensive": 1, "Harassment": 2}

# --- Training Hyperparameters (Centralized for easy logging) ---
TRAIN_ARGS = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
}

# --- Supervisor Report Tracking Variables ---
fold_results = [] 
all_loss_histories = {}
best_precision = 0.0 
champion_fold = -1

# --- 2. Load Data ---
def load_raw_data():
    print(f"📂 Loading data from: {DATA_DIR}")
    path_harass = os.path.join(DATA_DIR, "processed_consolidated_harassment.xlsx")
    path_offen = os.path.join(DATA_DIR, "processed_consolidated_offensive.xlsx")
    path_norm  = os.path.join(DATA_DIR, "processed_consolidated_normal.xlsx")

    df_harass = pd.read_excel(path_harass)
    df_offen = pd.read_excel(path_offen)
    df_norm  = pd.read_excel(path_norm)

    df_norm['label'], df_offen['label'], df_harass['label'] = 0, 1, 2
    df = pd.concat([df_harass, df_offen, df_norm], ignore_index=True)
    df = df[['cleaned_text', 'label']].dropna()
    df.rename(columns={'cleaned_text': 'text'}, inplace=True)
    df['text'] = df['text'].astype(str)
    print(f"✅ Loaded {len(df)} rows successfully.")
    return df

# --- 3. Helper to Balance a Training Split ---
def balance_training_data(train_df):
    df_norm = train_df[train_df['label'] == 0]
    df_offen = train_df[train_df['label'] == 1]
    df_harass = train_df[train_df['label'] == 2]
    target_count = len(df_norm)
    df_harass_upsampled = resample(df_harass, replace=True, n_samples=target_count, random_state=42)
    df_offen_upsampled = resample(df_offen, replace=True, n_samples=target_count, random_state=42)
    df_balanced = pd.concat([df_norm, df_offen_upsampled, df_harass_upsampled])
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# --- 4. Print Config ---
print("\n" + "="*60)
print("⚙️ CONFIGURATION: DISK SAVING MODE (SHARDING DISABLED)")
print("="*60)
for k, v in TRAIN_ARGS.items():
    print(f"{k:<30}: {v}")
print("="*60 + "\n")

df = load_raw_data()
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- 5. Cross Validation Loop ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n{'='*20} FOLD {fold + 1}/{N_FOLDS} {'='*20}")
    train_df_raw = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    train_df_balanced = balance_training_data(train_df_raw)

    train_dataset = Dataset.from_pandas(train_df_balanced).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
    
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id
    ).to("cuda")
    
    # --- NO SHARD SAVING ARGUMENTS ---
    training_args = TrainingArguments(
        output_dir=f'./results/fold_{fold}',
        num_train_epochs=TRAIN_ARGS["num_train_epochs"],
        per_device_train_batch_size=TRAIN_ARGS["per_device_train_batch_size"], 
        per_device_eval_batch_size=TRAIN_ARGS["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAIN_ARGS["gradient_accumulation_steps"],
        learning_rate=TRAIN_ARGS["learning_rate"],
        warmup_steps=TRAIN_ARGS["warmup_steps"],
        weight_decay=TRAIN_ARGS["weight_decay"],
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="no",             # <--- DISABLED DISK WRITING
        load_best_model_at_end=False,   # <--- DISABLED FOR SPEED
        fp16=True,
        dataloader_num_workers=0,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model, args=training_args, 
        train_dataset=train_dataset, eval_dataset=val_dataset, 
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Extract loss history
    history = trainer.state.log_history
    epochs_data = {}
    for log in history:
        ep = round(log.get('epoch', 0))
        if ep == 0: continue
        if ep not in epochs_data:
            epochs_data[ep] = {'train_loss': 'N/A', 'eval_loss': 'N/A'}
        if 'loss' in log:
            epochs_data[ep]['train_loss'] = round(log['loss'], 4)
        if 'eval_loss' in log:
            epochs_data[ep]['eval_loss'] = round(log['eval_loss'], 4)
            
    all_loss_histories[fold + 1] = epochs_data

    # Final Evaluation for the fold
    eval_result = trainer.evaluate()
    p = eval_result['eval_precision']
    fold_results.append({
        'fold': fold + 1,
        'accuracy': eval_result['eval_accuracy'],
        'precision': p,
        'recall': eval_result['eval_recall'],
        'f1': eval_result['eval_f1']
    })
    
    if p > best_precision:
        best_precision = p
        champion_fold = fold + 1
        
    print(f"Fold {fold+1} Metrics -> Acc: {eval_result['eval_accuracy']:.4f} | Prec: {p:.4f} | Rec: {eval_result['eval_recall']:.4f} | F1: {eval_result['eval_f1']:.4f}")

    del model, trainer; gc.collect(); torch.cuda.empty_cache()

# =====================================================================
# 6. SUPERVISOR REPORT EXTRACTION
# =====================================================================
print("\n" + "="*60)
print("📊 1. DATA FOR LOSS CURVES (OVERFITTING PROOF)")
print("="*60)
for f, losses in all_loss_histories.items():
    print(f"\n[ FOLD {f} LOSS HISTORY ]")
    print(f"{'Epoch':<10} | {'Train Loss':<15} | {'Eval Loss':<15}")
    print("-" * 45)
    for ep, data in sorted(losses.items()):
        print(f"{ep:<10} | {data['train_loss']:<15} | {data['eval_loss']:<15}")

print("\n" + "="*60)
print("📈 2. DATA FOR PERFORMANCE METRICS")
print("="*60)
for res in fold_results:
    print(f"Fold {res['fold']} -> Accuracy: {res['accuracy']:.4f} | Precision (W): {res['precision']:.4f} | Recall (W): {res['recall']:.4f} | F1-Score (W): {res['f1']:.4f}")

print("\n" + "="*60)
print("🏆 3. SELECTION LOGIC")
print("="*60)
print(f"Precision Champion Score : {best_precision:.4f} (Achieved in Fold {champion_fold})")

avg_acc = np.mean([r['accuracy'] for r in fold_results])
avg_prec = np.mean([r['precision'] for r in fold_results])
avg_f1 = np.mean([r['f1'] for r in fold_results])

print("\n" + "="*60)
print("🎯 4. FINAL AVERAGE SCORES")
print("="*60)
print(f"AVERAGE Accuracy   : {avg_acc:.4f}")
print(f"AVERAGE Precision  : {avg_prec:.4f}")
print(f"AVERAGE F1-Score   : {avg_f1:.4f}")
print("="*60)