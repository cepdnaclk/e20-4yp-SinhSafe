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
N_FOLDS = 5 
id2label = {0: "Normal", 1: "Offensive", 2: "Cyberbullying"}
label2id = {"Normal": 0, "Offensive": 1, "Cyberbullying": 2}

# --- 1. Load Data (RAW & UNBALANCED) ---
# We load data here but DO NOT oversample yet to prevent leakage
def load_raw_data():
    df_bully = pd.read_excel(os.path.join(DATA_DIR, "processed_cyberbullying.xlsx"))
    df_offen = pd.read_excel(os.path.join(DATA_DIR, "processed_offensive.xlsx"))
    df_norm = pd.read_excel(os.path.join(DATA_DIR, "processed_normal.xlsx"))

    df_norm['label'] = 0
    df_offen['label'] = 1
    df_bully['label'] = 2

    df = pd.concat([df_bully, df_offen, df_norm], ignore_index=True)
    df = df[['cleaned_text', 'label']].dropna()
    df.rename(columns={'cleaned_text': 'text'}, inplace=True)
    df['text'] = df['text'].astype(str)
    return df

# --- 2. Helper to Balance a Training Split ---
def balance_training_data(train_df):
    # Separate classes
    df_norm = train_df[train_df['label'] == 0]
    df_offen = train_df[train_df['label'] == 1]
    df_bully = train_df[train_df['label'] == 2]

    # Target count is the majority class in this specific fold
    target_count = len(df_norm)

    # Upsample minority classes
    df_bully_upsampled = resample(df_bully, replace=True, n_samples=target_count, random_state=42)
    df_offen_upsampled = resample(df_offen, replace=True, n_samples=target_count, random_state=42)

    # Combine
    df_balanced = pd.concat([df_norm, df_offen_upsampled, df_bully_upsampled])
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle

# --- Setup ---
df = load_raw_data() # Load raw data
print(f"Total Raw Data: {len(df)}")
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# --- DEBUG: Sanity Check ---
print("\n--- TOKENIZER CHECK ---")
sample_text = df['text'].iloc[0]  # Grab the first sentence
print(f"Original: {sample_text}")
print(f"Tokens:   {tokenizer.tokenize(sample_text)}")
print("-----------------------\n")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- Cross Validation Loop ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_results = []

print(f"Starting {N_FOLDS}-Fold Cross-Validation on RTX 3090...")

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    
    # A. Split Raw Data (Pure Split)
    train_df_raw = df.iloc[train_idx]
    val_df = df.iloc[val_idx] # Validation data is NEVER touched/balanced
    
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
        num_train_epochs=5,               # Increased to 5 (Safe bet)
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
        print(">> New Best Model! Saving...")
        model.save_pretrained("models/best_cv_model")
        tokenizer.save_pretrained("models/best_cv_model")

    # H. Cleanup
    del model, trainer, training_args
    gc.collect()
    torch.cuda.empty_cache()

# --- Final Report ---
print("\n" + "="*30)
print(f"Final Results ({N_FOLDS}-Fold CV)")
print(f"Average Accuracy: {np.mean(fold_results):.4f}")
print("="*30)