import pandas as pd
import numpy as np
import torch
import gc
import os
import shutil
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- Configuration ---
MODEL_NAME = "xlm-roberta-large"
DATA_DIR = r"data/processed_ground_truth"
SAVE_MODEL_DIR = "models/xlm_v1_model"
N_FOLDS = 5 

# EFFECTIVE BATCH SIZE = 16
# 4 (Physical) * 4 (Accumulation) = 16
PHYSICAL_BATCH_SIZE = 4 
ACCUMULATION_STEPS = 4 

id2label = {0: "Normal", 1: "Offensive", 2: "Harassment"}
label2id = {"Normal": 0, "Offensive": 1, "Harassment": 2}

# --- 1. Custom Architecture with MLP Head ---
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
        return {"logits": logits}

# --- 2. Custom Weighted Trainer ---
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Added Label Smoothing for better precision generalization
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        loss = loss_fct(logits.view(-1, 3), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- 3. Data Loading ---
def load_raw_data():
    df_harass = pd.read_excel(os.path.join(DATA_DIR, "processed_consolidated_harassment.xlsx"))
    df_offen = pd.read_excel(os.path.join(DATA_DIR, "processed_consolidated_offensive.xlsx"))
    df_norm  = pd.read_excel(os.path.join(DATA_DIR, "processed_consolidated_normal.xlsx"))

    df_norm['label'], df_offen['label'], df_harass['label'] = 0, 1, 2
    df = pd.concat([df_harass, df_offen, df_norm], ignore_index=True)
    df = df[['cleaned_text', 'label']].dropna()
    df.rename(columns={'cleaned_text': 'text'}, inplace=True)
    return df

df = load_raw_data()
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = logits.argmax(-1)
    # Using Macro to prioritize hard categories
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- 4. 5-Fold Cross Validation ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_precisions = []
best_precision = 0.0 

print(f"🚀 Starting XLM-R Large | Effective Batch 16 | Precision Focus")

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n--- FOLD {fold + 1} ---")
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    class_counts = train_df['label'].value_counts().sort_index().values
    weights = 1.0 / class_counts
    current_weights = torch.tensor(weights / weights.sum(), dtype=torch.float).to("cuda")

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
    
    model = SinhSafeClassifier.from_pretrained(MODEL_NAME, num_labels=3).to("cuda")
    
    training_args = TrainingArguments(
        output_dir=f'./results/xlm_fold_{fold}',
        num_train_epochs=5,
        per_device_train_batch_size=PHYSICAL_BATCH_SIZE, 
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        learning_rate=1e-5,
        weight_decay=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="precision", # Changed to Precision
        greater_is_better=True,
        save_total_limit=1,
        fp16=True,
        report_to="none"
    )
    
    trainer = WeightedTrainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        compute_metrics=compute_metrics,
        class_weights=current_weights
    )
    
    trainer.train()
    eval_result = trainer.evaluate()
    
    p = eval_result['eval_precision']
    print(f"Fold {fold+1} Final Precision: {p:.4f}")
    fold_precisions.append(p)
    
    if p > best_precision:
        best_precision = p
        model.save_pretrained(SAVE_MODEL_DIR)
        print(f"🌟 Best Precision Model Updated!")

    shutil.rmtree(f'./results/xlm_fold_{fold}', ignore_errors=True)
    del model, trainer; gc.collect(); torch.cuda.empty_cache()

print(f"\nFINAL AVG PRECISION: {np.mean(fold_precisions):.4f} (+/- {np.std(fold_precisions):.4f})")