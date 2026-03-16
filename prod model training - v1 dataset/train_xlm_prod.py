import os
# --- GPU TARGETING (Parallel Lane) ---
# Keeping this on GPU 2 so it doesn't interfere with SinLlama!
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import shutil
from torch import nn
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
MODEL_NAME = "xlm-roberta-large"

# --- DATA PATH (Local) ---
current_dir = os.getcwd()
DATA_DIR = os.path.join(current_dir, "data", "processed_ground_truth")

# --- MODEL & TEMP PATHS (Strictly Scratch Drive) ---
SCRATCH_PATH = "/scratch1/e20-4yp-sinhsafe"
SAVE_MODEL_DIR = os.path.join(SCRATCH_PATH, "xlm_prod_model")
TEMP_DIR = os.path.join(SCRATCH_PATH, "temp_xlm_results")

# Strict Label Mapping
id2label = {0: "Normal", 1: "Offensive", 2: "Harassment"}
label2id = {"Normal": 0, "Offensive": 1, "Harassment": 2}

# ==========================================
# 2. CUSTOM ARCHITECTURE
# ==========================================
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
        return {"loss": loss, "logits": logits}

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 3. DATA LOADING (100% Data)
# ==========================================
def load_raw_data():
    path_harass = os.path.join(DATA_DIR, "processed_consolidated_harassment.xlsx")
    path_offen = os.path.join(DATA_DIR, "processed_consolidated_offensive.xlsx")
    path_norm  = os.path.join(DATA_DIR, "processed_consolidated_normal.xlsx")

    df_harass = pd.read_excel(path_harass)
    df_offen = pd.read_excel(path_offen)
    df_norm  = pd.read_excel(path_norm)

    # Applying the strict label mapping
    df_norm['label'] = 0 
    df_offen['label'] = 1
    df_harass['label'] = 2
    
    df = pd.concat([df_harass, df_offen, df_norm], ignore_index=True)
    df = df[['cleaned_text', 'label']].dropna()
    df.rename(columns={'cleaned_text': 'text'}, inplace=True)
    df['text'] = df['text'].astype(str)
    return df

# ==========================================
# 4. MAIN EXECUTION (No CV, Just Full Train)
# ==========================================
def run_production_training():
    print("🚀 Starting 100% Data Production Training on RTX 6000 Ada...")
    
    df = load_raw_data()
    print(f"📊 Loaded {len(df)} total rows. No validation split—using ALL data.")

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    # Calculate Weights for Imbalance
    class_counts = df['label'].value_counts().sort_index().values
    weights = len(df) / (3.0 * class_counts)
    current_weights = torch.tensor(weights, dtype=torch.float).to("cuda")
    print(f"⚖️ Class Weights Applied: {weights}")

    # Create Single Dataset
    full_dataset = Dataset.from_pandas(df).map(tokenize_function, batched=True)
    full_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = SinhSafeClassifier.from_pretrained(
        MODEL_NAME, 
        num_labels=3, 
        id2label=id2label, 
        label2id=label2id
    ).to("cuda")

    # ADA-Optimized Training Arguments
    training_args = TrainingArguments(
        output_dir=TEMP_DIR,             # Safely mapped to Scratch
        num_train_epochs=4,              # Locked at 4 to prevent overfitting
        per_device_train_batch_size=16,  # Increased for Ada 48GB VRAM
        gradient_accumulation_steps=2,   # Effective batch size 32
        learning_rate=1e-5,
        warmup_steps=150,
        weight_decay=0.05,
        eval_strategy="no",              # No eval set to check against
        save_strategy="no",              # Only save at the very end
        fp16=True,
        dataloader_num_workers=4,        # Faster data loading on the new server
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model, 
        args=training_args, 
        train_dataset=full_dataset,
        class_weights=current_weights
    )

    print(f"\n🔥 Firing up the trainer (Outputting temp files to {TEMP_DIR})... See you at Epoch 4!")
    trainer.train()

    print(f"\n🌟 Training Complete! Saving God Model to {SAVE_MODEL_DIR}...")
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    model.save_pretrained(SAVE_MODEL_DIR)
    tokenizer.save_pretrained(SAVE_MODEL_DIR)
    
    print("🧹 Cleaning up temp directory...")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    print(f"🎉 Production Model successfully built and saved safely in {SCRATCH_PATH}!")

if __name__ == "__main__":
    run_production_training()