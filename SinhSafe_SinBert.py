import os
import shutil

# --- 1. CRITICAL: GPU TARGETING ---
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
from torch.amp import autocast, GradScaler 

# --- 2. CONFIGURATION & PRODUCTION PATHS ---
MODEL_NAME = "NLPC-UOM/SinBERT-large"
MAX_LEN = 128
PHYSICAL_BATCH_SIZE = 16
ACCUMULATION_STEPS = 1  

# 🎯 Locked to 2 Epochs based on CV optimal results
EPOCHS = 2 
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 💾 Scratch Storage Path
SCRATCH_PATH = "/scratch1/e20-4yp-sinhsafe"
OUTPUT_DIR = os.path.join(SCRATCH_PATH, "models", "sinbert_production")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. Custom Dataset Class ---
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

# --- 4. Custom SinBERT Architecture ---
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

# --- 5. Training Function ---
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

# --- 6. Main Production Logic ---
def run_production_training(df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"\n🚀 Starting PRODUCTION Training on 100% Data ({len(df)} rows)...")
    print(f"🎯 Target: {EPOCHS} Epochs")

    # 100% of data goes into the Train Loader
    train_ds = SinhSafeDataset(df['cleaned_text'].values, df['label'].values, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True)

    model = SinBERTClassifier(n_classes=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        print(f"\n{'='*20} Epoch {epoch+1}/{EPOCHS} {'='*20}")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler)
        print(f"✅ Epoch {epoch+1} Completed | Final Train Loss: {train_loss:.4f}")

    # Save the final production model to the Scratch Path
    print(f"\n💾 Saving Final Production Model to Scratch Drive...")
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "sinbert_production_model.bin"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"🌟 Success! Production Model is safely stored at:\n{os.path.abspath(OUTPUT_DIR)}")

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
    run_production_training(df)