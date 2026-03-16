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
SAVE_MODEL_DIR = "models/xlm_v3_model"
N_FOLDS = 5 

id2label = {0: "Normal", 1: "Offensive", 2: "Harassment"}
label2id = {"Normal": 0, "Offensive": 1, "Harassment": 2}

# --- 1. Custom Architecture with MLP Head ---
class SinhSafeClassifier(XLMRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # Using a Sequential classifier to replace the default head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 1. Get base model outputs
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 2. Extract the [CLS] token (first token of every sequence)
        sequence_output = outputs[0] 
        cls_token_state = sequence_output[:, 0, :] 

        # 3. Pass through your custom MLP head
        logits = self.classifier(cls_token_state) 

        # 4. Standard return format expected by HF Trainer
        loss = None
        if labels is not None:
            # We let the WeightedTrainer handle the actual loss calculation below,
            # but returning it in this format keeps the API happy.
            pass
            
        return {"loss": loss, "logits": logits}

# --- 2. Custom Weighted Trainer ---
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        
        # Handle dict output from our custom forward
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
        
        # Compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# --- 3. Data Loading ---
def load_raw_data():
    path_harass = os.path.join(DATA_DIR, "processed_consolidated_harassment.xlsx")
    path_offen = os.path.join(DATA_DIR, "processed_consolidated_offensive.xlsx")
    path_norm  = os.path.join(DATA_DIR, "processed_consolidated_normal.xlsx")

    df_harass = pd.read_excel(path_harass)
    df_offen = pd.read_excel(path_offen)
    df_norm  = pd.read_excel(path_norm)

    df_norm['label'], df_offen['label'], df_harass['label'] = 0, 1, 2
    df = pd.concat([df_harass, df_offen, df_norm], ignore_index=True)
    df = df[['cleaned_text', 'label']].dropna()
    df.rename(columns={'cleaned_text': 'text'}, inplace=True)
    df['text'] = df['text'].astype(str)
    return df

df = load_raw_data()
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

def compute_metrics(pred):
    labels = pred.label_ids
    # Handle dict vs tuple outputs safely
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = logits.argmax(-1)
        
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- 4. 5-Fold Cross Validation ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_accuracies, fold_precisions, fold_f1_scores = [], [], []
best_precision = 0.0 

print(f"🚀 Starting 5-Fold CV on Turing (RTX 3090 Ti)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n{'='*20} FOLD {fold + 1} {'='*20}")
    
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    # Weights calculation
    class_counts = train_df['label'].value_counts().sort_index().values
    weights = len(train_df) / (3.0 * class_counts)
    current_weights = torch.tensor(weights, dtype=torch.float).to("cuda")

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    model = SinhSafeClassifier.from_pretrained(MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id).to("cuda")
    
    # ADJUSTED: Keep disk clean and optimize for Turing
    training_args = TrainingArguments(
        output_dir=f'./results/fold_{fold}',
        num_train_epochs=8,
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4, # Effective batch remains 32
        learning_rate=1e-5,
        warmup_steps=150,
        weight_decay=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,     # Reverts to best weights at end of training
        metric_for_best_model="eval_loss",
        greater_is_better=False,         # Required because lower loss is better
        save_total_limit=1,              # CRITICAL: Keeps only 1 checkpoint at a time
        fp16=True,
        dataloader_num_workers=0,        # Stability on Turing
        report_to="none"
    )
    
    trainer = WeightedTrainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        compute_metrics=compute_metrics,
        class_weights=current_weights    # Pass weights during init
    )
    
    trainer.train()
    eval_result = trainer.evaluate()
    
    acc, prec, f1 = eval_result['eval_accuracy'], eval_result['eval_precision'], eval_result['eval_f1']
    print(f"Fold {fold+1} Metrics -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")
    
    fold_accuracies.append(acc)
    fold_precisions.append(prec)
    fold_f1_scores.append(f1)
    
    if prec > best_precision:
        best_precision = prec
        print(f"🌟 New High Precision! Saving Model...")
        model.save_pretrained(SAVE_MODEL_DIR)
        tokenizer.save_pretrained(SAVE_MODEL_DIR)

    # --- Disk Space Cleanup ---
    print(f"🧹 Cleaning up disk space for Fold {fold+1}...")
    shutil.rmtree(f'./results/fold_{fold}', ignore_errors=True)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "X"*40)
print(f"FINAL CV RESULTS: Precision: {np.mean(fold_precisions):.4f} (+/- {np.std(fold_precisions):.4f})")