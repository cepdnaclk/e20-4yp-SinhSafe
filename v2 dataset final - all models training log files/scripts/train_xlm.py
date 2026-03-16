import os
import shutil

# --- 1. CRITICAL: GPU TARGETING ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import gc
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- 2. Configuration & Paths ---
MODEL_NAME = "xlm-roberta-large"
DATA_DIR = os.path.join(os.getcwd(), "data", "processed_ground_truth", "v2")
SCRATCH_PATH = "/scratch1/e20-4yp-sinhsafe"
SAVE_MODEL_DIR = os.path.join(SCRATCH_PATH, "models", "xlm_run2")
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

N_FOLDS = 5 

id2label = {0: "Normal", 1: "Offensive", 2: "Harassment"}
label2id = {"Normal": 0, "Offensive": 1, "Harassment": 2}

# --- Tracking Variables for Supervisor Report ---
fold_results = [] 
all_loss_histories = {}
best_precision = 0.0 
champion_fold = -1

# --- 3. Custom Architecture with MLP Head ---
class SinhSafeClassifier(XLMRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0] 
        cls_token_state = sequence_output[:, 0, :] 
        logits = self.classifier(cls_token_state) 

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            
        return {"loss": loss, "logits": logits}

# --- 4. Data Loading ---
def load_v2_data():
    path_harass = os.path.join(DATA_DIR, "processed_final_harassment.csv")
    path_offen  = os.path.join(DATA_DIR, "processed_final_offensive.csv")
    path_norm   = os.path.join(DATA_DIR, "processed_final_normal.csv")

    df_harass = pd.read_csv(path_harass)
    df_offen  = pd.read_csv(path_offen)
    df_norm   = pd.read_csv(path_norm)

    df_norm['label'], df_offen['label'], df_harass['label'] = 0, 1, 2
    df = pd.concat([df_harass, df_offen, df_norm], ignore_index=True)
    df = df[['cleaned_text', 'label']].dropna()
    df.rename(columns={'cleaned_text': 'text'}, inplace=True)
    df['text'] = df['text'].astype(str)
    
    print(f"✅ Loaded {len(df)} balanced rows from V2 data.")
    return df

df = load_v2_data()
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = logits.argmax(-1)
        
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- 5. 5-Fold Cross Validation ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print(f"🚀 Starting 5-Fold CV on RTX 6000 Ada (GPU 2)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n{'='*20} FOLD {fold + 1} {'='*20}")
    
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    model = SinhSafeClassifier.from_pretrained(MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id).to("cuda")
    fold_output_dir = os.path.join(SCRATCH_PATH, f'results/fold_{fold}')
    
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4, 
        learning_rate=1e-5,
        warmup_steps=150,
        weight_decay=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",        # Added to extract exact Train Loss per epoch
        load_best_model_at_end=True,     
        metric_for_best_model="eval_loss",
        greater_is_better=False,         
        save_total_limit=1,              
        fp16=True,
        dataloader_num_workers=0,        
        report_to="none"
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        compute_metrics=compute_metrics
    )
    
    # 1. Train the model
    trainer.train()
    
    # 2. Extract Loss Curves Data
    history = trainer.state.log_history
    fold_loss_data = []
    for i in range(1, int(training_args.num_train_epochs) + 1):
        t_loss, e_loss = None, None
        for log in history:
            if 'epoch' in log and round(log['epoch']) == i:
                if 'loss' in log: t_loss = log['loss']
                if 'eval_loss' in log: e_loss = log['eval_loss']
        fold_loss_data.append({'epoch': i, 'train_loss': t_loss, 'eval_loss': e_loss})
    all_loss_histories[fold + 1] = fold_loss_data

    # 3. Evaluate Best Model
    eval_result = trainer.evaluate()
    
    acc = eval_result['eval_accuracy']
    prec = eval_result['eval_precision']
    f1 = eval_result['eval_f1']
    recall = eval_result['eval_recall']
    
    print(f"Fold {fold+1} Metrics -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    
    fold_results.append({
        'fold': fold + 1,
        'accuracy': acc,
        'precision': prec,
        'recall': recall,
        'f1': f1
    })
    
    if prec > best_precision:
        best_precision = prec
        champion_fold = fold + 1
        print(f"🌟 New High Precision! Saving Model to SCRATCH...")
        model.save_pretrained(SAVE_MODEL_DIR)
        tokenizer.save_pretrained(SAVE_MODEL_DIR)

    print(f"🧹 Cleaning up intermediate scratch files for Fold {fold+1}...")
    shutil.rmtree(fold_output_dir, ignore_errors=True)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
# 6. REPORT EXTRACTION (Prints perfectly formatted data for the user)
# =====================================================================
print("\n" + "="*60)
print("📊 1. DATA FOR LOSS CURVES (OVERFITTING PROOF)")
print("="*60)
for f, losses in all_loss_histories.items():
    print(f"\n[ FOLD {f} LOSS HISTORY ]")
    print(f"{'Epoch':<10} | {'Train Loss':<15} | {'Eval Loss':<15}")
    print("-" * 45)
    for log in losses:
        t_loss = f"{log['train_loss']:.4f}" if log['train_loss'] is not None else "N/A"
        e_loss = f"{log['eval_loss']:.4f}" if log['eval_loss'] is not None else "N/A"
        print(f"{log['epoch']:<10} | {t_loss:<15} | {e_loss:<15}")

print("\n" + "="*60)
print("📈 2. DATA FOR PERFORMANCE METRICS (BEST EPOCH PER FOLD)")
print("="*60)
for res in fold_results:
    print(f"Fold {res['fold']} -> Accuracy: {res['accuracy']:.4f} | Precision (W): {res['precision']:.4f} | Recall (W): {res['recall']:.4f} | F1-Score (W): {res['f1']:.4f}")

print("\n" + "="*60)
print("🏆 3. SELECTION LOGIC")
print("="*60)
print(f"Champion Fold           : Fold {champion_fold}")
print(f"Precision Champion Score: {best_precision:.4f}")
print(f"Winning Model Path      : {os.path.abspath(SAVE_MODEL_DIR)}")

print("\n" + "="*60)
print("🎯 4. FINAL AVERAGE SCORES")
print("="*60)
avg_acc = np.mean([r['accuracy'] for r in fold_results])
avg_prec = np.mean([r['precision'] for r in fold_results])
avg_rec = np.mean([r['recall'] for r in fold_results])
avg_f1 = np.mean([r['f1'] for r in fold_results])

print(f"AVERAGE Accuracy   : {avg_acc:.4f}")
print(f"AVERAGE Precision  : {avg_prec:.4f}")
print(f"AVERAGE Recall     : {avg_rec:.4f}")
print(f"AVERAGE F1-Score   : {avg_f1:.4f}")
print("="*60)