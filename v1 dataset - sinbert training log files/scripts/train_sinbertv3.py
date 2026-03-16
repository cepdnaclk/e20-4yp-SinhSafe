import os
# --- 1. CRITICAL: GPU TARGETING ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import gc
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW  
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import autocast, GradScaler 

# --- Configuration ---
TRAIN_ARGS = {
    "model_name": "NLPC-UOM/SinBERT-large",
    "max_len": 128,
    "physical_batch_size": 32,
    "accumulation_steps": 2,  # Effective Batch Size = 64
    "epochs": 8,
    "learning_rate": 2e-5,
    "dropout_p": 0.3
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Custom Dataset Class ---
class SinhSafeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

# --- 3. Custom SinBERT Architecture ---
class SinBERTClassifier(nn.Module):
    def __init__(self, n_classes, dropout_p=TRAIN_ARGS["dropout_p"]):
        super(SinBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(TRAIN_ARGS["model_name"])
        hidden_size = self.bert.config.hidden_size 
        self.lstm = nn.LSTM(hidden_size, 512, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(512 * 2 * 2, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        avg_pool = torch.mean(lstm_out, 1)
        max_pool, _ = torch.max(lstm_out, 1)
        combined = torch.cat((avg_pool, max_pool), dim=1) 
        return self.classifier(self.dropout(combined))

# --- 4. Training Function with Mixed Precision ---
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, scaler):
    model.train()
    losses = []
    optimizer.zero_grad()
    
    for step, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(DEVICE)
        attention_mask = d["attention_mask"].to(DEVICE)
        labels = d["labels"].to(DEVICE)

        with autocast(device_type='cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels) / TRAIN_ARGS["accumulation_steps"]

        scaler.scale(loss).backward()

        if (step + 1) % TRAIN_ARGS["accumulation_steps"] == 0 or (step + 1) == len(data_loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
        losses.append(loss.item() * TRAIN_ARGS["accumulation_steps"])
            
    return np.mean(losses)

# --- 5. Main Execution Logic ---
def run_cv_training(df):
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_ARGS["model_name"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Supervisor tracking variables
    fold_results, all_loss_histories = [], {}
    best_overall_prec, champion_fold = 0.0, -1

    print(f"🚀 Starting 5-Fold CV on GPU 1 (No-Sharding Mode)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['cleaned_text'], df['label'])):
        print(f"\n{'='*20} FOLD {fold + 1} {'='*20}")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_ds = SinhSafeDataset(train_df['cleaned_text'].values, train_df['label'].values, tokenizer, TRAIN_ARGS["max_len"])
        val_ds = SinhSafeDataset(val_df['cleaned_text'].values, val_df['label'].values, tokenizer, TRAIN_ARGS["max_len"])

        train_loader = DataLoader(train_ds, batch_size=TRAIN_ARGS["physical_batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=TRAIN_ARGS["physical_batch_size"])

        model = SinBERTClassifier(n_classes=3).to(DEVICE)
        
        class_counts = train_df['label'].value_counts().sort_index().values
        weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=weights/weights.sum())

        optimizer = AdamW(model.parameters(), lr=TRAIN_ARGS["learning_rate"], weight_decay=0.01)
        total_steps = (len(train_loader) // TRAIN_ARGS["accumulation_steps"]) * TRAIN_ARGS["epochs"]
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
        scaler = GradScaler()

        epochs_data = {}
        best_fold_prec = 0.0
        best_fold_metrics = {}

        for epoch in range(TRAIN_ARGS["epochs"]):
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler)
            
            model.eval()
            val_preds, val_labels, val_losses = [], [], []
            with torch.no_grad():
                for d in val_loader:
                    ids, mask, labs = d["input_ids"].to(DEVICE), d["attention_mask"].to(DEVICE), d["labels"].to(DEVICE)
                    with autocast(device_type='cuda'):
                        out = model(ids, mask)
                        v_loss = loss_fn(out, labs)
                    
                    val_losses.append(v_loss.item())
                    val_preds.extend(torch.max(out, dim=1)[1].cpu().numpy())
                    val_labels.extend(labs.cpu().numpy())

            avg_v_loss = np.mean(val_losses)
            acc = accuracy_score(val_labels, val_preds)
            prec, rec, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted', zero_division=0)
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Eval Loss: {avg_v_loss:.4f} | Prec: {prec:.4f} | F1: {f1:.4f}")
            epochs_data[epoch+1] = {'train_loss': round(train_loss, 4), 'eval_loss': round(avg_v_loss, 4)}

            if prec > best_fold_prec:
                best_fold_prec = prec
                best_fold_metrics = {'fold': fold+1, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
                if prec > best_overall_prec:
                    best_overall_prec = prec
                    champion_fold = fold + 1

        fold_results.append(best_fold_metrics)
        all_loss_histories[fold + 1] = epochs_data
        
        # Cleanup fold memory
        del model, optimizer; gc.collect(); torch.cuda.empty_cache()

    # --- FINAL SUPERVISOR REPORT ---
    print("\n" + "="*60 + "\n📊 1. DATA FOR LOSS CURVES (OVERFITTING PROOF)\n" + "="*60)
    for f, losses in all_loss_histories.items():
        print(f"\n[ FOLD {f} LOSS HISTORY ]\n{'Epoch':<10} | {'Train Loss':<15} | {'Eval Loss':<15}\n" + "-"*45)
        for ep, data in sorted(losses.items()):
            print(f"{ep:<10} | {data['train_loss']:<15.4f} | {data['eval_loss']:<15.4f}")

    print("\n" + "="*60 + "\n📈 2. PERFORMANCE METRICS (BEST EPOCH PER FOLD)\n" + "="*60)
    for res in fold_results:
        print(f"Fold {res['fold']} -> Acc: {res['accuracy']:.4f} | Prec(W): {res['precision']:.4f} | Rec(W): {res['recall']:.4f} | F1(W): {res['f1']:.4f}")

    print("\n" + "="*60 + "\n🎯 3. FINAL AVERAGE SCORES\n" + "="*60)
    avg_prec = np.mean([r['precision'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    print(f"AVERAGE Precision: {avg_prec:.4f} | AVERAGE F1: {avg_f1:.4f}\n" + "="*60)

if __name__ == "__main__":
    print("\n" + "="*60 + "\n⚙️ CONFIG: SINBERT V3 | NO SHARDS | AMP ENABLED\n" + "="*60)
    for k, v in TRAIN_ARGS.items(): print(f"{k:<30}: {v}")
    print("="*60 + "\nLoading data...")
    df_h = pd.read_excel('./data/processed_ground_truth/processed_consolidated_harassment.xlsx')
    df_o = pd.read_excel('./data/processed_ground_truth/processed_consolidated_offensive.xlsx')
    df_n = pd.read_excel('./data/processed_ground_truth/processed_consolidated_normal.xlsx')
    df_n['label'], df_o['label'], df_h['label'] = 0, 1, 2
    df = pd.concat([df_h, df_o, df_n], ignore_index=True).dropna(subset=['cleaned_text', 'label'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    run_cv_training(df)