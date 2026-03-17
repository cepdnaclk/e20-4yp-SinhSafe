import pandas as pd
import numpy as np
import torch
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.utils import resample
import os

# --- Configuration ---
MODEL_NAME = "xlm-roberta-large"
DATA_DIR = r"data/processed_ground_truth"
SAVE_MODEL_DIR = "models/best_xlm_roberta"  # <--- NEW NAME HERE
N_FOLDS = 5 

id2label = {0: "Normal", 1: "Offensive", 2: "Harassment"}
label2id = {"Normal": 0, "Offensive": 1, "Harassment": 2}

# --- 1. Load Data (RAW & UNBALANCED) ---
def load_raw_data():
    print(f"📂 Loading data from: {DATA_DIR}")
    
    path_harass = os.path.join(DATA_DIR, "processed_consolidated_harassment.xlsx")
    path_offen = os.path.join(DATA_DIR, "processed_consolidated_offensive.xlsx")
    path_norm  = os.path.join(DATA_DIR, "processed_consolidated_normal.xlsx")

    # Debug: Check if files exist
    if not os.path.exists(path_harass): raise FileNotFoundError(f"Missing: {path_harass}")
    if not os.path.exists(path_offen): raise FileNotFoundError(f"Missing: {path_offen}")
    if not os.path.exists(path_norm):  raise FileNotFoundError(f"Missing: {path_norm}")

    df_harass = pd.read_excel(path_harass)
    df_offen = pd.read_excel(path_offen)
    df_norm  = pd.read_excel(path_norm)

    # Assign Numeric Labels
    # 0 = Normal, 1 = Offensive, 2 = Harassment
    df_norm['label'] = 0
    df_offen['label'] = 1
    df_harass['label'] = 2

    # Combine
    df = pd.concat([df_harass, df_offen, df_norm], ignore_index=True)
    
    # Cleaning
    df = df[['cleaned_text', 'label']].dropna()
    df.rename(columns={'cleaned_text': 'text'}, inplace=True)
    df['text'] = df['text'].astype(str)
    
    print(f"✅ Loaded {len(df)} rows successfully.")
    return df

# --- 2. Helper to Balance a Training Split ---
def balance_training_data(train_df):
    # Separate classes
    df_norm = train_df[train_df['label'] == 0]
    df_offen = train_df[train_df['label'] == 1]
    df_harass = train_df[train_df['label'] == 2]

    target_count = len(df_norm)

    # Upsample minority classes
    df_harass_upsampled = resample(df_harass, replace=True, n_samples=target_count, random_state=42)
    df_offen_upsampled = resample(df_offen, replace=True, n_samples=target_count, random_state=42)

    # Combine
    df_balanced = pd.concat([df_norm, df_offen_upsampled, df_harass_upsampled])
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Setup ---
df = load_raw_data()
print(f"Total Raw Data: {len(df)}")

print("🔄 Loading Tokenizer...")
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# --- DEBUG: Sanity Check ---
print("\n--- TOKENIZER CHECK ---")
sample_text = df['text'].iloc[0]
print(f"Original: {sample_text}")
print(f"Tokens:   {tokenizer.tokenize(sample_text)}")
print("-----------------------\n")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- Cross Validation Loop ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_results = []

print(f"🚀 Starting {N_FOLDS}-Fold Cross-Validation on RTX 3090...")

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    
    # A. Split Raw Data
    train_df_raw = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # B. Balance ONLY Training Data
    train_df_balanced = balance_training_data(train_df_raw)
    
    print(f"   Train Size (Balanced): {len(train_df_balanced)} | Val Size (Original): {len(val_df)}")

    # C. Create Datasets
    train_dataset = Dataset.from_pandas(train_df_balanced)
    val_dataset = Dataset.from_pandas(val_df)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # D. Initialize Model
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id
    ).to("cuda")
    
    # E. Training Args
    training_args = TrainingArguments(
        output_dir=f'./results/fold_{fold}',
        num_train_epochs=10,
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model, args=training_args, 
        train_dataset=train_dataset, eval_dataset=val_dataset, 
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # F. Evaluate
    eval_result = trainer.evaluate()
    print(f"Fold {fold+1} Accuracy: {eval_result['eval_accuracy']:.4f}")
    fold_results.append(eval_result['eval_accuracy'])
    
    # G. Save Best
    if eval_result['eval_accuracy'] == max(fold_results):
        print(f">> New Best Model! Saving to {SAVE_MODEL_DIR}...")
        model.save_pretrained(SAVE_MODEL_DIR)
        tokenizer.save_pretrained(SAVE_MODEL_DIR)

    # H. Cleanup
    del model, trainer, training_args
    gc.collect()
    torch.cuda.empty_cache()

# --- Final Report ---
print("\n" + "="*30)
print(f"Final Results ({N_FOLDS}-Fold CV)")
print(f"Average Accuracy: {np.mean(fold_results):.4f}")
print("="*30)