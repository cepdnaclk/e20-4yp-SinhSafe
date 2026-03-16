import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import json
import gc
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW  
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.amp import autocast, GradScaler 

# --- Configuration ---
MODEL_NAME = "NLPC-UOM/SinBERT-large"
MAX_LEN = 128
PHYSICAL_BATCH_SIZE = 16
ACCUMULATION_STEPS = 1  
EPOCHS = 4
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./models/sinbert_v4_batch16"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Custom Dataset Class ---
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

# --- 2. Custom SinBERT Architecture ---
class SinBERTClassifier(nn.Module):
    def __init__(self, n_classes, dropout_p=0.3):
        super(SinBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
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

# --- 3. Training Function ---
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
            loss = loss_fn(outputs, labels)
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(data_loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
        losses.append(loss.item() * ACCUMULATION_STEPS)
            
    return np.mean(losses)

# --- 4. Main Execution Logic ---
def run_cv_training(df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    label_mapping = {"Normal": 0, "Offensive": 1, "Harassment": 2}
    
    all_fold_results = []
    best_overall_precision = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['cleaned_text'], df['label'])):
        print(f"\n{'='*20} Starting Fold {fold+1} {'='*20}")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_ds = SinhSafeDataset(train_df['cleaned_text'].values, train_df['label'].values, tokenizer, MAX_LEN)
        val_ds = SinhSafeDataset(val_df['cleaned_text'].values, val_df['label'].values, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=PHYSICAL_BATCH_SIZE)

        model = SinBERTClassifier(n_classes=3).to(DEVICE)
        
        # Weighted Loss for Class Imbalance
        class_counts = train_df['label'].value_counts().sort_index().values
        weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(DEVICE)
        weights = weights / weights.sum()
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
        scaler = GradScaler()

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler)
            
            # Validation
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

            # Metrics
            avg_v_loss = np.mean(val_losses)
            prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
            rec = recall_score(val_labels, val_preds, average='macro')
            f1 = f1_score(val_labels, val_preds, average='macro')
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {avg_v_loss:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")

            if prec > best_overall_precision:
                torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_sinbert_model.bin")
                best_overall_precision = prec
                print(f"🌟 New Precision Champion Saved! (P: {best_overall_precision:.4f})")

        all_fold_results.append({'f1': f1, 'p': prec, 'r': rec})
        del model, optimizer; gc.collect(); torch.cuda.empty_cache()

    # Final Summary
    print("\n" + "X"*50 + "\n🏆 FINAL CROSS-VALIDATION SUMMARY\n" + "X"*50)
    avg_p = np.mean([x['p'] for x in all_fold_results])
    avg_f1 = np.mean([x['f1'] for x in all_fold_results])
    print(f"Average Precision: {avg_p:.4f}")
    print(f"Average Macro F1:  {avg_f1:.4f}")

if __name__ == "__main__":
    df_h = pd.read_excel('./data/processed_ground_truth/processed_consolidated_harassment.xlsx')
    df_o = pd.read_excel('./data/processed_ground_truth/processed_consolidated_offensive.xlsx')
    df_n = pd.read_excel('./data/processed_ground_truth/processed_consolidated_normal.xlsx')
    df_n['label'], df_o['label'], df_h['label'] = 0, 1, 2
    df = pd.concat([df_h, df_o, df_n], ignore_index=True).dropna(subset=['cleaned_text', 'label'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    run_cv_training(df)