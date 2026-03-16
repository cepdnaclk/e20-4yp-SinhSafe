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

# --- 1. CRITICAL: GPU TARGETING ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
MODEL_NAME = "xlm-roberta-large"
DATA_DIR = r"data/processed_ground_truth"
N_FOLDS = 5 

id2label = {0: "Normal", 1: "Offensive", 2: "Harassment"}
label2id = {"Normal": 0, "Offensive": 1, "Harassment": 2}

# --- Training Hyperparameters ---
TRAIN_ARGS = {
    "num_train_epochs": 8,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2, 
    "learning_rate": 1e-5,
    "warmup_steps": 150,
    "weight_decay": 0.05,
}

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
        return {"loss": None, "logits": logits}

# --- 3. Custom Weighted Trainer ---
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

# --- 4. Data Loading ---
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
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- 5. Print Config ---
print("\n" + "="*60)
print("⚙️ TRAINING CONFIGURATION (DISK SAVING MODE ENABLED)")
print("="*60)
for k, v in TRAIN_ARGS.items():
    print(f"{k:<30}: {v}")
print("="*60 + "\n")

# --- 6. 5-Fold Cross Validation ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n{'='*20} FOLD {fold + 1} {'='*20}")
    
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    class_counts = train_df['label'].value_counts().sort_index().values
    weights = len(train_df) / (3.0 * class_counts)
    current_weights = torch.tensor(weights, dtype=torch.float).to("cuda")

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    model = SinhSafeClassifier.from_pretrained(MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id).to("cuda")
    
    # --- NO SHARD SAVING ARGUMENTS ---
    training_args = TrainingArguments(
        output_dir=f'./results/fold_{fold}',
        num_train_epochs=TRAIN_ARGS["num_train_epochs"],
        per_device_train_batch_size=TRAIN_ARGS["per_device_train_batch_size"], 
        per_device_eval_batch_size=TRAIN_ARGS["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAIN_ARGS["gradient_accumulation_steps"],
        learning_rate=TRAIN_ARGS["learning_rate"],
        warmup_steps=150,
        weight_decay=0.05,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="no",             # <--- DISABLED DISK WRITING
        load_best_model_at_end=False,   # <--- DISABLED FOR SPEED
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
    
    # Extract data for graphs
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

    # Evaluate current epoch (Last epoch)
    eval_result = trainer.evaluate()
    acc, prec, rec, f1 = eval_result['eval_accuracy'], eval_result['eval_precision'], eval_result['eval_recall'], eval_result['eval_f1']
    
    fold_results.append({'fold': fold + 1, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
    if prec > best_precision:
        best_precision = prec
        champion_fold = fold + 1

    del model, trainer; gc.collect(); torch.cuda.empty_cache()

# --- Supervisor Report Output ---
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
print("📈 2. DATA FOR PERFORMANCE METRICS")
print("="*60)
for res in fold_results:
    print(f"Fold {res['fold']} -> Accuracy: {res['accuracy']:.4f} | Precision (W): {res['precision']:.4f} | Recall (W): {res['recall']:.4f} | F1-Score (W): {res['f1']:.4f}")

avg_prec = np.mean([r['precision'] for r in fold_results])
avg_f1 = np.mean([r['f1'] for r in fold_results])
print(f"\nAVERAGE Precision: {avg_prec:.4f} | AVERAGE F1: {avg_f1:.4f}")
print("="*60)