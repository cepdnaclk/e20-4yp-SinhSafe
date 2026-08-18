import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.metrics import f1_score
from sklearn.utils import resample

# --- 1. CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Using your preferred free GPU
SINBERT_PATH = "/scratch1/e20-4yp-sinhsafe/sinbert_prod_model_v2"
XLM_PATH = "/scratch1/e20-4yp-sinhsafe/xlm_prod_model_95"
TEST_DATA_PATH = "test_10.csv"
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. RECREATE CUSTOM ARCHITECTURES ---
class SinBERTClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout_p=0.3):
        super(SinBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("NLPC-UOM/SinBERT-large")
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

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0] 
        cls_token_state = sequence_output[:, 0, :] 
        logits = self.classifier(cls_token_state) 
        return logits

# --- 3. LOAD MODELS ---
print("📂 Loading Custom Production Models...")
sinbert_tokenizer = AutoTokenizer.from_pretrained(SINBERT_PATH)
sinbert_model = SinBERTClassifier(n_classes=3).to(DEVICE)
sinbert_model.load_state_dict(torch.load(os.path.join(SINBERT_PATH, "best_sinbert_model.bin")))
sinbert_model.eval()

xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(XLM_PATH)
xlm_model = SinhSafeClassifier.from_pretrained(XLM_PATH).to(DEVICE)
xlm_model.eval()

# --- 4. DATA PREPARATION (Bulletproof Mapping) ---
print("📊 Loading Unseen Test Dataset (test_10.csv)...")
df_test = pd.read_csv(TEST_DATA_PATH)

label_map = {
    "normal": 0, "offensive": 1, "harassment": 2,
    "0": 0, "1": 1, "2": 2
}
df_test['label'] = df_test['label'].astype(str).str.strip().str.lower().map(label_map)
df_test = df_test.dropna(subset=['cleaned_text', 'label'])
df_test['label'] = df_test['label'].astype(int)

texts = df_test['cleaned_text'].astype(str).tolist()
true_labels = df_test['label'].tolist()

print(f"✅ Successfully loaded {len(texts)} pristine test samples.")

# --- 5. INFERENCE PASS (Batched for Speed/Safety) ---
print("🔍 Running Full Suite Inference...")
sinbert_preds = []
xlm_preds = []
ensemble_preds = []

with torch.no_grad():
    # Chunk texts into batches to maximize GPU efficiency and avoid deadlocks
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Evaluating"):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        # Tokenize
        sin_inputs = sinbert_tokenizer(batch_texts, max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(DEVICE)
        xlm_inputs = xlm_tokenizer(batch_texts, max_length=256, padding='max_length', truncation=True, return_tensors='pt').to(DEVICE)

        # Forward Pass
        sin_logits = sinbert_model(sin_inputs['input_ids'], sin_inputs['attention_mask'])
        xlm_logits = xlm_model(input_ids=xlm_inputs['input_ids'], attention_mask=xlm_inputs['attention_mask'])

        # Probabilities
        sin_probs = F.softmax(sin_logits, dim=1).cpu().numpy()
        xlm_probs = F.softmax(xlm_logits, dim=1).cpu().numpy()

        # Iterate through the batch to apply your custom ensemble logic
        for j in range(len(batch_texts)):
            s_prob = sin_probs[j]
            x_prob = xlm_probs[j]
            
            # Solo Predictions
            sinbert_preds.append(np.argmax(s_prob))
            xlm_preds.append(np.argmax(x_prob))

            # Ensemble Logic (FIXED TYPO: > 0.90)
           if s_prob[2] > 0.90 or x_prob[2] > 0.90:
                ensemble_preds.append(2)
            else:
                avg_probs = (s_prob + x_prob) / 2
                ensemble_preds.append(np.argmax(avg_probs))'''
            avg_probs = (s_prob + x_prob) / 2
            ensemble_preds.append(np.argmax(avg_probs)) '''

# --- 6. BOOTSTRAPPING FOR INDEPENDENT 95% CI ---
print("📈 Calculating Independent 95% Confidence Intervals (1000 iterations)...")

def calculate_ci(y_true, y_pred, n_iterations=1000):
    bootstrapped_scores = []
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    for _ in range(n_iterations):
        indices = resample(np.arange(len(y_true)), replace=True)
        score = f1_score(y_true_np[indices], y_pred_np[indices], average='macro')
        bootstrapped_scores.append(score)
        
    base_f1 = f1_score(y_true, y_pred, average='macro')
    lower = np.percentile(bootstrapped_scores, 2.5)
    upper = np.percentile(bootstrapped_scores, 97.5)
    return base_f1, lower, upper

sin_base, sin_l, sin_u = calculate_ci(true_labels, sinbert_preds)
xlm_base, xlm_l, xlm_u = calculate_ci(true_labels, xlm_preds)
ens_base, ens_l, ens_u = calculate_ci(true_labels, ensemble_preds)

print("\n" + "="*70)
print("🏆 FINAL TEST SET RESULTS (MACRO F1-SCORE)")
print("="*70)
print(f"SinBERT (Custom)    : {sin_base * 100:.2f}% (95% CI: [{sin_l * 100:.2f}%, {sin_u * 100:.2f}%])")
print(f"XLM-RoBERTa (Custom): {xlm_base * 100:.2f}% (95% CI: [{xlm_l * 100:.2f}%, {xlm_u * 100:.2f}%])")
print(f"SinhSafe Ensemble   : {ens_base * 100:.2f}% (95% CI: [{ens_l * 100:.2f}%, {ens_u * 100:.2f}%])")
print("="*70)


# --- 7. PAIRED BOOTSTRAP SIGNIFICANCE TEST ---
print("\n🔬 Running Paired Bootstrap Test (Ensemble vs XLM-RoBERTa)...")

def paired_bootstrap_test(y_true, preds_A, preds_B, n_iterations=1000):
    differences = []
    y_true_np = np.array(y_true)
    preds_A_np = np.array(preds_A) # Ensemble
    preds_B_np = np.array(preds_B) # XLM-RoBERTa
    
    for _ in range(n_iterations):
        # 1. Pick a random fake dataset (same indices for both models)
        indices = resample(np.arange(len(y_true)), replace=True)
        
        # 2. Calculate both scores on this exact same fake dataset
        score_A = f1_score(y_true_np[indices], preds_A_np[indices], average='macro')
        score_B = f1_score(y_true_np[indices], preds_B_np[indices], average='macro')
        
        # 3. Record the difference (Delta)
        differences.append(score_A - score_B)
        
    # 4. Calculate the 95% Confidence Interval of the DIFFERENCE
    lower_bound = np.percentile(differences, 2.5)
    upper_bound = np.percentile(differences, 97.5)
    mean_diff = np.mean(differences)
    
    return mean_diff, lower_bound, upper_bound

# Run the test comparing Ensemble to XLM-RoBERTa
mean_diff, lower, upper = paired_bootstrap_test(true_labels, ensemble_preds, xlm_preds)

print("\n" + "="*70)
print("⚖️ PAIRED SIGNIFICANCE TEST RESULTS")
print("="*70)
print(f"Average F1 Advantage of Ensemble: {mean_diff * 100:+.2f}%")
print(f"95% CI of the Difference: [{lower * 100:.2f}%, {upper * 100:.2f}%]")
print("-" * 70)

# The Golden Rule of Significance
if lower > 0:
    print("✅ CONCLUSION: STATISTICALLY SIGNIFICANT!")
    print("Because the interval is entirely above 0%, we can confidently state")
    print("that the Ensemble is significantly better than XLM-RoBERTa.")
else:
    print("❌ CONCLUSION: NOT STATISTICALLY SIGNIFICANT.")
    print("Because the interval crosses 0%, XLM-RoBERTa occasionally beat")
    print("the Ensemble. We cannot definitively claim the Ensemble is better.")
print("="*70)