import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW  
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import autocast, GradScaler

# --- Configuration ---
MODEL_NAME = "NLPC-UOM/SinBERT-large"
MAX_LEN = 128
PHYSICAL_BATCH_SIZE = 32
ACCUMULATION_STEPS = 2  # Effective Batch Size = 32
EPOCHS = 8
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./models/sinbert_v3"
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
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer(
            text,
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
            'labels': torch.tensor(label, dtype=torch.long)
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
        sequence_output = outputs.last_hidden_state 
        lstm_out, _ = self.lstm(sequence_output)
        
        avg_pool = torch.mean(lstm_out, 1)
        max_pool, _ = torch.max(lstm_out, 1)
        combined = torch.cat((avg_pool, max_pool), dim=1) 
        
        return self.classifier(self.dropout(combined))

# --- 3. Training Function (Optimized with FP16 & Accumulation) ---
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, scaler):
    model.train()
    losses = []
    
    for step, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(DEVICE)
        attention_mask = d["attention_mask"].to(DEVICE)
        labels = d["labels"].to(DEVICE)

        # Cast operations to mixed precision
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            # Normalize loss for accumulation
            loss = loss / ACCUMULATION_STEPS

        # Scales loss and calls backward to create scaled gradients
        scaler.scale(loss).backward()
        losses.append(loss.item() * ACCUMULATION_STEPS)

        # Update weights only after ACCUMULATION_STEPS
        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(data_loader):
            # Unscale before clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
    return np.mean(losses)

# --- 4. Main Execution Logic ---
def run_cv_training(df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Strict Label Mapping
    label_mapping = {"Normal": 0, "Offensive": 1, "Harassment": 2}
    
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map(label_mapping)
    elif set(df['label'].unique()) != {0, 1, 2}:
        df['label'] = df['label'].astype(int)
        
    print(f"Strict Label Mapping enforced: {label_mapping}")
    with open(f"{OUTPUT_DIR}/label_mapping.json", "w") as f:
        json.dump(label_mapping, f)

    # Class Weights setup
    class_counts = df['label'].value_counts().sort_index().values
    weights = 1.0 / class_counts
    weights = torch.tensor(weights / weights.sum(), dtype=torch.float).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # Trackers for Final Printout
    all_fold_f1s, all_fold_accs, all_fold_precs, all_fold_recs = [], [], [], []
    best_overall_f1 = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['cleaned_text'], df['label'])):
        print(f"\n{'='*20} Starting Fold {fold+1} {'='*20}")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_ds = SinhSafeDataset(train_df['cleaned_text'].values, train_df['label'].values, tokenizer, MAX_LEN)
        val_ds = SinhSafeDataset(val_df['cleaned_text'].values, val_df['label'].values, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=PHYSICAL_BATCH_SIZE)

        model = SinBERTClassifier(n_classes=len(class_counts)).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
        
        # Initialize FP16 Scaler
        scaler = GradScaler()

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler)
            
            # Validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for d in val_loader:
                    ids, mask, labels = d["input_ids"].to(DEVICE), d["attention_mask"].to(DEVICE), d["labels"].to(DEVICE)
                    outputs = model(ids, mask)
                    _, preds = torch.max(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            fold_f1 = f1_score(val_labels, val_preds, average='macro')
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Macro F1: {fold_f1:.4f}")

        # Final Evaluation at the end of the Fold
        final_fold_acc = accuracy_score(val_labels, val_preds)
        final_fold_prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        final_fold_rec = recall_score(val_labels, val_preds, average='macro')
        final_fold_f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"\n--- Fold {fold+1} Final Metrics ---")
        print(f"Accuracy:  {final_fold_acc:.4f}")
        print(f"Precision: {final_fold_prec:.4f}")
        print(f"Recall:    {final_fold_rec:.4f}")
        print(f"Macro F1:  {final_fold_f1:.4f}")
        
        all_fold_accs.append(final_fold_acc)
        all_fold_precs.append(final_fold_prec)
        all_fold_recs.append(final_fold_rec)
        all_fold_f1s.append(final_fold_f1)

        if final_fold_f1 > best_overall_f1:
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_sinbert_model.bin")
            best_overall_f1 = final_fold_f1
            print(f"🌟 New Overall Best Model Saved! (Macro F1: {best_overall_f1:.4f})")
            
    # --- Final CV Summary ---
    print("\n" + "X"*50)
    print("🏆 FINAL 5-FOLD CROSS VALIDATION RESULTS")
    print("X"*50)
    print(f"Average Accuracy:  {np.mean(all_fold_accs):.4f} (+/- {np.std(all_fold_accs):.4f})")
    print(f"Average Precision: {np.mean(all_fold_precs):.4f} (+/- {np.std(all_fold_precs):.4f})")
    print(f"Average Recall:    {np.mean(all_fold_recs):.4f} (+/- {np.std(all_fold_recs):.4f})")
    print(f"Average Macro F1:  {np.mean(all_fold_f1s):.4f} (+/- {np.std(all_fold_f1s):.4f})")
    print("X"*50)

if __name__ == "__main__":
    print("Loading data...")
    # 1. Load raw files
    df_harass = pd.read_excel('./data/processed_ground_truth/processed_consolidated_harassment.xlsx')
    df_offen = pd.read_excel('./data/processed_ground_truth/processed_consolidated_offensive.xlsx')
    df_norm  = pd.read_excel('./data/processed_ground_truth/processed_consolidated_normal.xlsx')

    # 2. Assign Numeric Labels immediately (Best Practice)
    df_norm['label'], df_offen['label'], df_harass['label'] = 0, 1, 2
    
    # 3. Combine and Clean (Exactly like your XLM script)
    df = pd.concat([df_harass, df_offen, df_norm], ignore_index=True)
    df = df[['cleaned_text', 'label']].dropna() # The safety net!
    
    # 4. Final Cleanup
    df['cleaned_text'] = df['cleaned_text'].astype(str)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total rows loaded: {len(df)}")
    run_cv_training(df)