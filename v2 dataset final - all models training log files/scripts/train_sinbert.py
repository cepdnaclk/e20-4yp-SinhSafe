import os
# --- 1. CRITICAL: GPU TARGETING ---
# Target GPU 2 (RTX 6000 Ada) to avoid the memory hogs on 0 and 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.amp import autocast, GradScaler 

# --- Configuration & Paths ---
MODEL_NAME = "NLPC-UOM/SinBERT-large"
MAX_LEN = 128
PHYSICAL_BATCH_SIZE = 16
ACCUMULATION_STEPS = 1  
EPOCHS = 4
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save directly to the current directory's 'models' folder
OUTPUT_DIR = os.path.join(os.getcwd(), "models", "sinbert_v2_balanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    def __init__(self, n_classes, dropout_p=0.3):
        super(SinBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size 
        
        # Adding a Bidirectional LSTM on top of BERT for sequential context
        self.lstm = nn.LSTM(hidden_size, 512, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(512 * 2 * 2, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        
        # Concatenating Average Pooling and Max Pooling
        avg_pool = torch.mean(lstm_out, 1)
        max_pool, _ = torch.max(lstm_out, 1)
        combined = torch.cat((avg_pool, max_pool), dim=1) 
        
        return self.classifier(self.dropout(combined))

# --- 4. Training Function ---
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

# --- 5. Main Execution Logic ---
def run_cv_training(df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- Tracking Variables for Supervisor Report ---
    all_fold_results = []
    all_loss_histories = {}
    best_overall_precision = 0.0
    champion_fold = -1

    print(f"🚀 Starting 5-Fold CV on RTX 6000 Ada (GPU 2)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['cleaned_text'], df['label'])):
        print(f"\n{'='*20} Starting Fold {fold+1} {'='*20}")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_ds = SinhSafeDataset(train_df['cleaned_text'].values, train_df['label'].values, tokenizer, MAX_LEN)
        val_ds = SinhSafeDataset(val_df['cleaned_text'].values, val_df['label'].values, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=PHYSICAL_BATCH_SIZE)

        model = SinBERTClassifier(n_classes=3).to(DEVICE)
        
        # STANDARD LOSS: Dataset is balanced, so no weights needed!
        loss_fn = nn.CrossEntropyLoss()

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
        scaler = GradScaler()

        best_fold_prec = 0.0
        best_fold_metrics = {}
        fold_loss_data = []  # To track loss curve data

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
            acc = accuracy_score(val_labels, val_preds)
            prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
            rec = recall_score(val_labels, val_preds, average='macro')
            f1 = f1_score(val_labels, val_preds, average='macro')

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {avg_v_loss:.4f} | Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")

            # Store data for the Loss Curve
            fold_loss_data.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'eval_loss': avg_v_loss
            })

            # Track best metrics for this specific fold
            if prec > best_fold_prec:
                best_fold_prec = prec
                best_fold_metrics = {'acc': acc, 'f1': f1, 'p': prec, 'r': rec}

            # Save ONLY the single best model overall (overwrites previous best)
            if prec > best_overall_precision:
                best_overall_precision = prec
                champion_fold = fold + 1
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_sinbert_model.bin"))
                tokenizer.save_pretrained(OUTPUT_DIR)
                print(f"🌟 New Precision Champion Saved! (P: {best_overall_precision:.4f})")

        # Append the *best* metrics and loss histories from this fold
        all_fold_results.append(best_fold_metrics)
        all_loss_histories[fold + 1] = fold_loss_data
        
        # Aggressive memory cleanup before next fold
        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    # =====================================================================
    # 6. SUPERVISOR REPORT EXTRACTION
    # =====================================================================
    report = "\n" + "="*60 + "\n"
    report += "📊 1. DATA FOR LOSS CURVES (OVERFITTING PROOF)\n"
    report += "="*60 + "\n"
    for f, losses in all_loss_histories.items():
        report += f"\n[ FOLD {f} LOSS HISTORY ]\n"
        report += f"{'Epoch':<10} | {'Train Loss':<15} | {'Eval Loss':<15}\n"
        report += "-" * 45 + "\n"
        for data in losses:
            report += f"{data['epoch']:<10} | {data['train_loss']:<15.4f} | {data['eval_loss']:<15.4f}\n"

    report += "\n" + "="*60 + "\n"
    report += "📈 2. DATA FOR PERFORMANCE METRICS (BEST EPOCH PER FOLD)\n"
    report += "="*60 + "\n"
    for i, res in enumerate(all_fold_results):
        report += f"Fold {i+1} -> Accuracy: {res['acc']:.4f} | Precision (W): {res['p']:.4f} | Recall (W): {res['r']:.4f} | F1-Score (W): {res['f1']:.4f}\n"

    report += "\n" + "="*60 + "\n"
    report += "🏆 3. SELECTION LOGIC\n"
    report += "="*60 + "\n"
    report += f"Precision Champion Score : {best_overall_precision:.4f} (Achieved in Fold {champion_fold})\n"
    report += f"Winning Model Path       : {os.path.abspath(OUTPUT_DIR)}\n"

    report += "\n" + "="*60 + "\n"
    report += "🎯 4. FINAL AVERAGE SCORES\n"
    report += "="*60 + "\n"
    avg_acc = np.mean([x['acc'] for x in all_fold_results])
    avg_p = np.mean([x['p'] for x in all_fold_results])
    avg_r = np.mean([x['r'] for x in all_fold_results])
    avg_f1 = np.mean([x['f1'] for x in all_fold_results])

    report += f"AVERAGE Accuracy   : {avg_acc:.4f}\n"
    report += f"AVERAGE Precision  : {avg_p:.4f}\n"
    report += f"AVERAGE Recall     : {avg_r:.4f}\n"
    report += f"AVERAGE F1-Score   : {avg_f1:.4f}\n"
    report += "="*60 + "\n"

    # Print to console and save to text file
    print(report)
    with open("sinbert_final_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ Clean report successfully saved to {os.path.abspath('sinbert_final_report.txt')}")


# --- 7. Load V2 CSV Data ---
if __name__ == "__main__":
    DATA_DIR = os.path.join(os.getcwd(), "data", "processed_ground_truth", "v2")
    path_harass = os.path.join(DATA_DIR, "processed_final_harassment.csv")
    path_offen  = os.path.join(DATA_DIR, "processed_final_offensive.csv")
    path_norm   = os.path.join(DATA_DIR, "processed_final_normal.csv")

    df_h = pd.read_csv(path_harass)
    df_o = pd.read_csv(path_offen)
    df_n = pd.read_csv(path_norm)
    
    df_n['label'], df_o['label'], df_h['label'] = 0, 1, 2
    
    df = pd.concat([df_h, df_o, df_n], ignore_index=True).dropna(subset=['cleaned_text', 'label'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} balanced rows from V2 data.")
    run_cv_training(df)