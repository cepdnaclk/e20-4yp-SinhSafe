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
SAVE_MODEL_DIR = os.path.join(SCRATCH_PATH, "xlm_prod_model_95")
TEMP_DIR = os.path.join(SCRATCH_PATH, "temp_xlm_results_95")

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
        
        # Loss is calculated externally by the WeightedTrainer
        return {"loss": None, "logits": logits}

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Hugging Face defaults to looking for "labels", ensure it exists
        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 3. DATA LOADING (90% DEVELOPMENT SET ONLY)
# ==========================================
def load_raw_data():
    # Load your pre-split 90% file
    print("📂 Loading the 90% Golden Training Split...")
    df = pd.read_csv("train_90.csv")
    
    # Ensure columns match the architecture expectation
    df = df.rename(columns={'cleaned_text': 'text'})
    df['text'] = df['text'].astype(str)
    
    # Map your labels if the CSV has string labels (Normal/Offensive/Harassment)
    if df['label'].dtype == 'object':
        df['label'] = df['label'].str.capitalize().map(label2id)
        
    return df

# ==========================================
# 4. MAIN EXECUTION (No CV, Just Full Train)
# ==========================================
def run_production_training():
    print("🚀 Starting 100% Data Production Training on RTX 6000 Ada...")
    
    df = load_raw_data()
    print(f"📊 Loaded {len(df)} total rows. No validation split—using ALL data.")

    # Calculate dynamic class weights for imbalanced datasets
    class_counts = df['label'].value_counts().sort_index().values
    total_samples = len(df)
    n_classes = len(id2label)
    computed_weights = total_samples / (n_classes * class_counts)
    current_weights = torch.tensor(computed_weights, dtype=torch.float).to("cuda")
    
    print(f"⚖️ Computed Class Weights: {computed_weights}")

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        # Keep labels as ints if already mapped, map if string
        labels = []
        for l in examples["label"]:
            if isinstance(l, str):
                normalized_label = l.capitalize()
                labels.append(label2id[normalized_label])
            else:
                labels.append(l)
            
        tokenized = tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=256
        )
        
        # Crucial Fix: HuggingFace Trainer expects 'labels' not 'label'
        tokenized["labels"] = labels 
        return tokenized

    # Create dataset directly from the mapped dataframe
    full_dataset = Dataset.from_pandas(df)
    full_dataset = full_dataset.map(tokenize_function, batched=True)

    # Explicitly set the format mapping to 'labels'
    full_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

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
        per_device_train_batch_size=8,  # Increased for Ada 48GB VRAM
        gradient_accumulation_steps=2,   # Effective batch size 16
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