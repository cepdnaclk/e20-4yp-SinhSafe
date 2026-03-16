import os
import shutil

# --- 1. CRITICAL: GPU TARGETING ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import gc
from torch import nn
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- 2. Configuration & Paths ---
MODEL_NAME = "xlm-roberta-large"
DATA_DIR = os.path.join(os.getcwd(), "data", "processed_ground_truth", "v2")

# 💾 Scratch Storage Path
SCRATCH_PATH = "/scratch1/e20-4yp-sinhsafe"
OUTPUT_DIR = os.path.join(SCRATCH_PATH, "models", "xlm_production")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🎯 Locked to 3 Epochs based on CV optimal results
EPOCHS = 3

id2label = {0: "Normal", 1: "Offensive", 2: "Harassment"}
label2id = {"Normal": 0, "Offensive": 1, "Harassment": 2}

# --- 3. Custom Architecture with MLP Head ---
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
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            
        return {"loss": loss, "logits": logits}

# --- 4. Data Loading (100% of Data) ---
def load_v2_data():
    path_harass = os.path.join(DATA_DIR, "processed_final_harassment.csv")
    path_offen  = os.path.join(DATA_DIR, "processed_final_offensive.csv")
    path_norm   = os.path.join(DATA_DIR, "processed_final_normal.csv")

    df_harass = pd.read_csv(path_harass)
    df_offen  = pd.read_csv(path_offen)
    df_norm   = pd.read_csv(path_norm)

    df_norm['label'], df_offen['label'], df_harass['label'] = 0, 1, 2
    df = pd.concat([df_harass, df_offen, df_norm], ignore_index=True)
    df = df[['cleaned_text', 'label']].dropna()
    df.rename(columns={'cleaned_text': 'text'}, inplace=True)
    df['text'] = df['text'].astype(str)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} balanced rows from V2 data.")
    return df

df = load_v2_data()
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

# --- 5. Production Training Logic ---
print(f"\n🚀 Starting PRODUCTION Training on 100% Data ({len(df)} rows)...")
print(f"🎯 Target: {EPOCHS} Epochs on GPU 2")

# Create the unified dataset
full_dataset = Dataset.from_pandas(df).map(tokenize_function, batched=True)
full_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Initialize Model
model = SinhSafeClassifier.from_pretrained(MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id).to("cuda")

# Configure Training Arguments for pure training (No Eval)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8, 
    gradient_accumulation_steps=4, 
    learning_rate=1e-5,
    warmup_steps=150,
    weight_decay=0.05,
    logging_strategy="epoch",        
    save_strategy="no",              # We will manually save at the very end
    fp16=True,
    dataloader_num_workers=0,        
    report_to="none"
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=full_dataset, 
)

# Train the model
trainer.train()

# Save the final production model to the Scratch Path
print(f"\n💾 Saving Final Production Model to Scratch Drive...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"🌟 Success! XLM-RoBERTa-large Production Model is safely stored at:\n{os.path.abspath(OUTPUT_DIR)}")

# Cleanup
del model, trainer
gc.collect()
torch.cuda.empty_cache()