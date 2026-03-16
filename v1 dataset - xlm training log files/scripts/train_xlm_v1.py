import os
# --- 1. CRITICAL: GPU TARGETING ---
# Targeting the free GPU 1 as requested
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutil
import pandas as pd
import numpy as np
import torch
import gc
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset



# --- Configuration ---
MODEL_NAME = "xlm-roberta-large"
DATA_DIR = r"data/processed_ground_truth"
N_FOLDS = 5 

# EFFECTIVE BATCH SIZE = 16
PHYSICAL_BATCH_SIZE = 16
ACCUMULATION_STEPS = 1

id2label = {0: "Normal", 1: "Offensive", 2: "Harassment"}
label2id = {"Normal": 0, "Offensive": 1, "Harassment": 2}

# --- Tracking Variables for Supervisor Report ---
fold_results = [] 
all_loss_histories = {}
best_precision = 0.0 
champion_fold = -1

# --- 2. Custom Architecture with MLP Head ---
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
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            
        return {"loss": loss, "logits": logits}

# --- 3. Data Loading ---
def load_raw_data():
    # Loading V1 data from Excel files as per your directory structure
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
    # Using 'weighted' average as requested for the final metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- 4. 5-Fold Cross Validation ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print(f"🚀 Starting XLM-R V1 Training | Extracting Logs for Graphs")

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n{'='*20} FOLD {fold + 1} {'='*20}")
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
    
    model = SinhSafeClassifier.from_pretrained(MODEL_NAME, num_labels=3).to("cuda")
    
    training_args = TrainingArguments(
        output_dir=f'./results/xlm_v1_fold_{fold}',
        num_train_epochs=5,
        per_device_train_batch_size=PHYSICAL_BATCH_SIZE, 
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        learning_rate=1e-5,
        weight_decay=0.05,
        eval_strategy="epoch",
        save_strategy="no", # DO NOT SAVE CHECKPOINTS
        logging_strategy="epoch", # CRITICAL: Captures Training Loss per epoch
        fp16=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        compute_metrics=compute_metrics
    )
    
    trainer.train()

    # --- EXTRACT LOSS DATA FOR GRAPHS ---
    history = trainer.state.log_history
    epochs_data = {}
    for log in history:
        ep = round(log.get('epoch', 0))
        if ep == 0: continue
        if ep not in epochs_data:
            epochs_data[ep] = {'train_loss': 'N/A', 'eval_loss': 'N/A'}
        if 'loss' in log:
            epochs_data[ep]['train_loss'] = round(log['loss'], 4)
        if 'eval_loss' in log:
            epochs_data[ep]['eval_loss'] = round(log['eval_loss'], 4)
            
    all_loss_histories[fold + 1] = epochs_data

    # --- FINAL EVALUATION ---
    eval_result = trainer.evaluate()
    p = eval_result['eval_precision']
    
    fold_results.append({
        'fold': fold + 1,
        'accuracy': eval_result['eval_accuracy'],
        'precision': p,
        'recall': eval_result['eval_recall'],
        'f1': eval_result['eval_f1']
    })
    
    if p > best_precision:
        best_precision = p
        champion_fold = fold + 1

    # Cleanup intermediate result folders
    shutil.rmtree(f'./results/xlm_v1_fold_{fold}', ignore_errors=True)
    del model, trainer; gc.collect(); torch.cuda.empty_cache()

# =====================================================================
# 5. SUPERVISOR REPORT EXTRACTION (Final Log Output)
# =====================================================================
print("\n" + "="*60)
print("📊 1. DATA FOR LOSS CURVES (OVERFITTING PROOF)")
print("="*60)
for f, losses in all_loss_histories.items():
    print(f"\n[ FOLD {f} LOSS HISTORY ]")
    print(f"{'Epoch':<10} | {'Train Loss':<15} | {'Eval Loss':<15}")
    print("-" * 45)
    for ep, data in sorted(losses.items()):
        print(f"{ep:<10} | {data['train_loss']:<15} | {data['eval_loss']:<15}")

print("\n" + "="*60)
print("📈 2. DATA FOR PERFORMANCE METRICS (FOLD-BY-FOLD)")
print("="*60)
for res in fold_results:
    print(f"Fold {res['fold']} -> Accuracy: {res['accuracy']:.4f} | Precision (W): {res['precision']:.4f} | Recall (W): {res['recall']:.4f} | F1-Score (W): {res['f1']:.4f}")

print("\n" + "="*60)
print("🏆 3. SELECTION LOGIC")
print("="*60)
print(f"Precision Champion Score : {best_precision:.4f} (Achieved in Fold {champion_fold})")
print(f"Note: This best model configuration was previously saved in the V1 training run.")

print("\n" + "="*60)
print("🎯 4. FINAL AVERAGE SCORES (V1 DATA)")
print("="*60)
avg_acc = np.mean([r['accuracy'] for r in fold_results])
avg_prec = np.mean([r['precision'] for r in fold_results])
avg_rec = np.mean([r['recall'] for r in fold_results])
avg_f1 = np.mean([r['f1'] for r in fold_results])

print(f"AVERAGE Accuracy   : {avg_acc:.4f}")
print(f"AVERAGE Precision  : {avg_prec:.4f}")
print(f"AVERAGE Recall     : {avg_rec:.4f}")
print(f"AVERAGE F1-Score   : {avg_f1:.4f}")
print("="*60)