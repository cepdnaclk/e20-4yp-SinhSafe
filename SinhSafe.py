import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# --- 1. CRITICAL: GPU TARGETING ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Running on GPU 2

# --- 2. PATHS TO YOUR SAVED PRODUCTION MODELS ---
# Make sure these match exactly where you saved the final production runs!
SINBERT_PATH = "/scratch1/e20-4yp-sinhsafe/models/sinbert_production"
XLM_PATH = "/scratch1/e20-4yp-sinhsafe/models/xlm_production"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "Normal", 1: "Offensive", 2: "Harassment"}

# --- 3. RECREATE MODEL ARCHITECTURES ---
# A. SinBERT Architecture
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

# B. XLM-RoBERTa Architecture
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

# --- 4. LOAD MODELS AND TOKENIZERS ---
print("📂 Loading SinBERT Production Model...")
sinbert_tokenizer = AutoTokenizer.from_pretrained(SINBERT_PATH)
sinbert_model = SinBERTClassifier(n_classes=3).to(DEVICE)
sinbert_model.load_state_dict(torch.load(os.path.join(SINBERT_PATH, "sinbert_production_model.bin")))
sinbert_model.eval()

print("📂 Loading XLM-RoBERTa Production Model...")
xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(XLM_PATH)
xlm_model = SinhSafeClassifier.from_pretrained(XLM_PATH).to(DEVICE)
xlm_model.eval()

# --- 5. THE ENSEMBLE LOGIC ---
def predict_text(text):
    # 1. Prepare inputs
    sin_inputs = sinbert_tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(DEVICE)
    xlm_inputs = xlm_tokenizer(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        # 2. Get raw logits
        sin_logits = sinbert_model(sin_inputs['input_ids'], sin_inputs['attention_mask'])
        xlm_logits = xlm_model(input_ids=xlm_inputs['input_ids'], attention_mask=xlm_inputs['attention_mask'])

        # 3. Convert to Probabilities (Percentages)
        sin_probs = F.softmax(sin_logits, dim=1).squeeze().cpu().numpy()
        xlm_probs = F.softmax(xlm_logits, dim=1).squeeze().cpu().numpy()

    # Index 2 is always "Harassment"
    sin_harass_conf = sin_probs[2]
    xlm_harass_conf = xlm_probs[2]

    print("\n" + "-"*55)
    print(f"📝 Input: {text}")
    print(f"🤖 SinBERT Probs: [Norm: {sin_probs[0]:.2f}, Offen: {sin_probs[1]:.2f}, Harass: {sin_probs[2]:.2f}]")
    print(f"🌐 XLM-R Probs  : [Norm: {xlm_probs[0]:.2f}, Offen: {xlm_probs[1]:.2f}, Harass: {xlm_probs[2]:.2f}]")

    # --- 4. APPLY CUSTOM SUPERVISOR LOGIC ---
    
    # Rule 1: The 90% Override
    if sin_harass_conf > 0.90 or xlm_harass_conf > 0.90:
        final_class = "Harassment"
        final_confidence = max(sin_harass_conf, xlm_harass_conf) * 100 
        print("🚨 TRIGGER: High Confidence Harassment Override (>90%)")
    
    else:
        # Rule 2: Soft Voting (Average the probabilities)
        avg_probs = (sin_probs + xlm_probs) / 2
        print(f"⚖️ Averaged Probs: [Norm: {avg_probs[0]:.2f}, Offen: {avg_probs[1]:.2f}, Harass: {avg_probs[2]:.2f}]")
        
        # Rule 3: Pick the highest average
        final_id = np.argmax(avg_probs)
        final_class = LABELS[final_id]
        final_confidence = avg_probs[final_id] * 100

    print(f"🏆 FINAL PREDICTION: >> {final_class} << (Confidence: {final_confidence:.2f}%)")
    print("-" * 55)

# --- 6. INTERACTIVE TESTING ---
if __name__ == "__main__":
    print("\n✅ Ensemble Ready! Type 'exit' to stop.")
    while True:
        user_input = input("\nType a Sinhala/Singlish sentence: ")
        if user_input.lower() == 'exit':
            break
        predict_text(user_input)