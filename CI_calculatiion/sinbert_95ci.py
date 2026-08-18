import os
# --- GPU TARGETING ---
# GPU 2 is ready for action
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW  
from torch.amp import autocast, GradScaler 

# ==========================================
# 1. CONFIGURATION & PATHS (Strict v5 Settings)
# ==========================================
MODEL_NAME = "NLPC-UOM/SinBERT-large"
MAX_LEN = 128
PHYSICAL_BATCH_SIZE = 32  # LOCKED exactly to v5!
ACCUMULATION_STEPS = 1    
EPOCHS = 2                # Locked at 2 to stop overfitting
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA PATH (Local) ---
current_dir = os.getcwd()
DATA_DIR = os.path.join(current_dir, "data", "processed_ground_truth")

# --- MODEL PATH (Strictly Scratch Drive) ---
SCRATCH_PATH = "/scratch1/e20-4yp-sinhsafe"
SAVE_MODEL_DIR = os.path.join(SCRATCH_PATH, "sinbert_prod_model_95")

# ==========================================
# 2. CUSTOM DATASET
# ==========================================
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

# ==========================================
# 3. V5 SINBERT ARCHITECTURE
# ==========================================
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

# ==========================================
# 4. TRAINING FUNCTION (100% Data)
# ==========================================
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, scaler, epoch):
    model.train()
    losses = []
    optimizer.zero_grad()
    
    loop = tqdm(data_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)
    
    for step, d in enumerate(loop):
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
        loop.set_postfix(loss=np.mean(losses))
            
    return np.mean(losses)

def run_production_training():
    print("🚀 Starting Production Training (90% Split) for SinBERT...")
    
    # 1. Load Data (THE 90% SPLIT)
    df = pd.read_csv("train_90.csv")
    
    # --- BULLETPROOF LABEL MAPPING ---
    # Accounts for text, string numbers, and actual ints
    label_map = {
        "normal": 0, "offensive": 1, "harassment": 2,
        "0": 0, "1": 1, "2": 2
    }
    
    # Force lowercase string, strip invisible spaces, and map
    df['label'] = df['label'].astype(str).str.strip().str.lower().map(label_map)
        
    # Drop rows that failed mapping (NaNs) or have missing text
    df = df.dropna(subset=['cleaned_text', 'label'])
    
    # STRICTLY cast to integer so PyTorch doesn't crash
    df['label'] = df['label'].astype(int)
    # ---------------------------------
    
    print(f"📊 Loaded {len(df)} rows from train_90.csv. Training leak-free.")

    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_ds = SinhSafeDataset(df['cleaned_text'].values, df['label'].values, tokenizer, MAX_LEN)
    train_loader = DataLoader(full_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, num_workers=0)

    # 3. Model Setup
    model = SinBERTClassifier(n_classes=3).to(DEVICE)
    
    # 4. Weighted Loss
    class_counts = df['label'].value_counts().sort_index().values
    weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(DEVICE)
    weights = weights / weights.sum()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    print(f"⚖️ Class Weights: {weights.cpu().numpy()}")

    # 5. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    scaler = GradScaler()

    # 6. Training Loop
    print(f"\n🔥 Firing training loop for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler, epoch)
        print(f"✅ Epoch {epoch} completed. Loss: {train_loss:.4f}")

    # 7. Save the Model (V2)
    SAVE_DIR_V2 = os.path.join(SCRATCH_PATH, "sinbert_prod_model_v2")
    print(f"\n🌟 Saving SinBERT God Model V2 to {SAVE_DIR_V2}...")
    os.makedirs(SAVE_DIR_V2, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR_V2, "best_sinbert_model.bin"))
    tokenizer.save_pretrained(SAVE_DIR_V2)
    
    print(f"🎉 Production Model V2 built and secured.")

if __name__ == "__main__":
    run_production_training()