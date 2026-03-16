import os
# --- GPU TARGETING ---
# GPU 2 is ready for action
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
# 1. CONFIGURATION & PATHS (Strict v4 Settings)
# ==========================================
MODEL_NAME = "NLPC-UOM/SinBERT-large"
MAX_LEN = 128
PHYSICAL_BATCH_SIZE = 16  # LOCKED exactly to v4!
ACCUMULATION_STEPS = 1    
EPOCHS = 2                # Locked at 2 to stop overfitting
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA PATH (Local) ---
current_dir = os.getcwd()
DATA_DIR = os.path.join(current_dir, "data", "processed_ground_truth")

# --- MODEL PATH (Strictly Scratch Drive) ---
SCRATCH_PATH = "/scratch1/e20-4yp-sinhsafe"
SAVE_MODEL_DIR = os.path.join(SCRATCH_PATH, "sinbert_prod_model")

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
# 3. V4 SINBERT ARCHITECTURE
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
    print("🚀 Starting 100% Data Production Training for SinBERT on GPU 2...")
    
    # 1. Load Data
    path_harass = os.path.join(DATA_DIR, "processed_consolidated_harassment.xlsx")
    path_offen = os.path.join(DATA_DIR, "processed_consolidated_offensive.xlsx")
    path_norm  = os.path.join(DATA_DIR, "processed_consolidated_normal.xlsx")

    df_h = pd.read_excel(path_harass)
    df_o = pd.read_excel(path_offen)
    df_n = pd.read_excel(path_norm)

    # Apply Strict Labels
    df_n['label'], df_o['label'], df_h['label'] = 0, 1, 2
    df = pd.concat([df_h, df_o, df_n], ignore_index=True).dropna(subset=['cleaned_text', 'label'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"📊 Loaded {len(df)} total rows. Training on ALL data.")

    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_ds = SinhSafeDataset(df['cleaned_text'].values, df['label'].values, tokenizer, MAX_LEN)
    train_loader = DataLoader(full_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, num_workers=4)

    # 3. Model Setup
    model = SinBERTClassifier(n_classes=3).to(DEVICE)
    
    # 4. Weighted Loss for Imbalance
    class_counts = df['label'].value_counts().sort_index().values
    weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(DEVICE)
    weights = weights / weights.sum()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    print(f"⚖️ Class Weights Applied: {weights.cpu().numpy()}")

    # 5. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    scaler = GradScaler()

    # 6. Training Loop (Exactly 2 Epochs)
    print(f"\n🔥 Firing up the training loop for {EPOCHS} epochs with Batch Size {PHYSICAL_BATCH_SIZE}...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler, epoch)
        print(f"✅ Epoch {epoch} completed. Average Training Loss: {train_loss:.4f}")

    # 7. Save the Model
    print(f"\n🌟 Training Complete! Saving SinBERT God Model to {SAVE_MODEL_DIR}...")
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_MODEL_DIR, "best_sinbert_model.bin"))
    tokenizer.save_pretrained(SAVE_MODEL_DIR)
    
    print(f"🎉 Production Model successfully built and secured in {SCRATCH_PATH}!")

if __name__ == "__main__":
    run_production_training()