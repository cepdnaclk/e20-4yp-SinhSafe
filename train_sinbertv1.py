import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW  
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import os

# --- Configuration ---
MODEL_NAME = "NLPC-UOM/SinBERT-large"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./models/sinbert_best"
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
        
        # [FIX] Call the tokenizer directly instead of using .encode_plus
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
        
        # Dynamically get the hidden size
        hidden_size = self.bert.config.hidden_size 
        
        # Bi-LSTM for Context Extraction
        self.lstm = nn.LSTM(hidden_size, 512, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)
        
        # Update classifier input size to handle the concatenated (avg + max) pooling
        self.classifier = nn.Linear(512 * 2 * 2, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 
        lstm_out, _ = self.lstm(sequence_output)
        
        # Use both Average and Max pooling for richer context representation
        avg_pool = torch.mean(lstm_out, 1)
        max_pool, _ = torch.max(lstm_out, 1)
        combined = torch.cat((avg_pool, max_pool), dim=1) 
        
        return self.classifier(self.dropout(combined))

# --- 3. Training Function ---
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler):
    model.train()
    losses = []
    for d in data_loader:
        input_ids = d["input_ids"].to(DEVICE)
        attention_mask = d["attention_mask"].to(DEVICE)
        labels = d["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return np.mean(losses)

# --- 4. Main Execution Logic ---
def run_cv_training(df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Map string labels to numeric if they aren't already
    if df['label'].dtype == 'object':
        # Hardcoding it is safer so it NEVER changes between runs
        label_mapping = {"Normal": 0, "Offensive": 1, "Harassment": 2}
        df['label'] = df['label'].map(label_mapping)
        print(f"Label mapping created: {label_mapping}")
        
        # Save it to the models folder so your test script can read it later
        with open(f"{OUTPUT_DIR}/label_mapping.json", "w") as f:
            json.dump(label_mapping, f)

    class_counts = df['label'].value_counts().sort_index().values
    weights = 1.0 / class_counts
    weights = torch.tensor(weights / weights.sum(), dtype=torch.float).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_overall_f1 = 0

    # [UPDATE] Use 'cleaned_text' for the Stratified split
    for fold, (train_idx, val_idx) in enumerate(skf.split(df['cleaned_text'], df['label'])):
        print(f"\n--- Starting Fold {fold+1} ---")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # [UPDATE] Pass the 'cleaned_text' column to the dataset
        train_ds = SinhSafeDataset(train_df['cleaned_text'].values, train_df['label'].values, tokenizer, MAX_LEN)
        val_ds = SinhSafeDataset(val_df['cleaned_text'].values, val_df['label'].values, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = SinBERTClassifier(n_classes=len(class_counts)).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler)
            
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

            fold_f1 = f1_score(val_labels, val_preds, average='weighted')
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val F1: {fold_f1:.4f}")

            if fold_f1 > best_overall_f1:
                torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_sinbert_model.bin")
                best_overall_f1 = fold_f1
                print(f"--> New Overall Best Model Saved! (F1: {best_overall_f1:.4f})")

# --- Execution Block ---
if __name__ == "__main__":
    print("Loading data...")
    try:
        # Load your three processed Excel files
        df_harassment = pd.read_excel('./data/processed_ground_truth/processed_consolidated_harassment.xlsx')
        df_normal = pd.read_excel('./data/processed_ground_truth/processed_consolidated_normal.xlsx')
        df_offensive = pd.read_excel('./data/processed_ground_truth/processed_consolidated_offensive.xlsx')

        # Combine them into a single dataframe
        df = pd.concat([df_harassment, df_normal, df_offensive], ignore_index=True)
        
        # [UPDATE] Security check specifically looks for 'cleaned_text'
        if 'cleaned_text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"Expected columns 'cleaned_text' and 'label', but found: {df.columns.tolist()}")

        # Shuffle the dataframe to mix the classes
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Total rows loaded: {len(df)}")
        print("Class Distribution:")
        print(df['label'].value_counts())
        
        print("\nStarting 5-Fold Cross Validation...")
        run_cv_training(df)

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure you are running this script from the root directory 'SinhSafe'.")
    except Exception as e:
        print(f"An error occurred: {e}")