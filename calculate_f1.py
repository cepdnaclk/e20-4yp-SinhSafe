import pandas as pd
from sklearn.metrics import classification_report, f1_score
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from tqdm import tqdm
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "models/best_cv_model"
DATA_DIR = "data/processed_ground_truth"

# How many rows to check from each file? (Keep low for speed, higher for accuracy)
SAMPLE_SIZE = 50 

# Define files and their CORRECT labels (Must match model labels exactly)
FILES_AND_LABELS = {
    'processed_cyberbullying.xlsx': 'Cyberbullying',
    'processed_normal.xlsx': 'Normal',
    'processed_offensive.xlsx': 'Offensive'
}

# ==========================================
# 2. LOAD MODEL
# ==========================================
print("🔄 Loading Model...")
try:
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Auto-detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ==========================================
# 3. PREPARE DATA
# ==========================================
all_texts = []
all_true_labels = []

print(f"\n📂 Loading {SAMPLE_SIZE} random samples from each file...")

for filename, true_label in FILES_AND_LABELS.items():
    file_path = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            
            # 1. Try to find the cleaned text column first
            if 'cleaned_text' in df.columns:
                target_col = 'cleaned_text'
            elif 'text' in df.columns:
                target_col = 'text'
            else:
                # Fallback: Use the first column
                target_col = df.columns[0]

            # 2. Sample data (Get random rows)
            if len(df) > SAMPLE_SIZE:
                df = df.sample(n=SAMPLE_SIZE, random_state=42)
            
            # 3. Add to our lists
            # Convert to string just in case
            texts = df[target_col].astype(str).tolist()
            all_texts.extend(texts)
            all_true_labels.extend([true_label] * len(texts))
            
            print(f"   ✅ Added {len(texts)} rows from {filename} (Label: {true_label})")
            
        except Exception as e:
            print(f"   ⚠️ Error reading {filename}: {e}")
    else:
        print(f"   ❌ File not found: {filename}")

# ==========================================
# 4. RUN PREDICTIONS
# ==========================================
print("\n🧠 Running Predictions...")

predicted_labels = []

# Batch processing would be faster, but single loop is safer for debugging
for text in tqdm(all_texts, desc="Predicting"):
    # Tokenize
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128, 
        padding="max_length"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=-1).item()
        
    # Convert ID (0,1,2) to Label (Normal, etc.)
    pred_label = model.config.id2label[pred_idx]
    predicted_labels.append(pred_label)

# ==========================================
# 5. GENERATE REPORT
# ==========================================
print("\n" + "="*50)
print("       📊 MODEL PERFORMANCE REPORT       ")
print("="*50)

# Calculate Macro F1 (Balanced view) and Weighted F1 (Population view)
macro_f1 = f1_score(all_true_labels, predicted_labels, average='macro')
weighted_f1 = f1_score(all_true_labels, predicted_labels, average='weighted')

print(f"🏆 Macro F1 Score:    {macro_f1:.4f} (Best for balanced performance)")
print(f"⚖️ Weighted F1 Score: {weighted_f1:.4f}")
print("-" * 50)
print("Detailed Breakdown:")
print(classification_report(all_true_labels, predicted_labels))
print("="*50)