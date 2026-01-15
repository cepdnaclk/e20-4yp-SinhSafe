import pandas as pd
import torch
import torch.nn.functional as F
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import sys
import os
import re
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
current_dir = os.getcwd()
MODEL_PATH = os.path.join(current_dir, "models", "best_cv_model")

FILE_LIST = [
    'data/day6.xlsx', 
    'data/day7.xlsx', 
    'data/day8.xlsx', 
    'data/day9.xlsx', 
    'data/day10.xlsx'
]

OUTPUT_FILES = {
    'Cyberbullying': 'cyberbullying_model_predict.xlsx',
    'Normal': 'normal_model_predict.xlsx',
    'Offensive': 'offensive_model_predict.xlsx'
}

# ==========================================
# 2. SETUP HYBRID CLEANING SYSTEM
# ==========================================
# Add 'src' to path to load your custom modules
sys.path.append(os.path.join(current_dir, 'src'))

# A. Load Google API Logic
try:
    from process_ground_truth import clean_and_process
    print("✅ Loaded: Google Online Transliteration")
    USE_GOOGLE = True
except ImportError:
    print("⚠️ Warning: Could not find 'process_ground_truth.py'.")
    USE_GOOGLE = False

# B. Load Offline Logic
try:
    from offline_transliteration import OfflineConverter
    # Initialize the converter once (loads dictionaries)
    offline_converter = OfflineConverter()
    print("✅ Loaded: Offline Transliteration Backup")
    USE_OFFLINE = True
except ImportError:
    print("⚠️ Warning: Could not find 'offline_transliteration.py'.")
    USE_OFFLINE = False

def smart_clean_text(text):
    """
    3-Layer Cleaning Strategy:
    1. Try Google API (High Accuracy)
    2. If fails, use Offline Converter (Medium Accuracy)
    3. If fails, use Regex (Basic Safety)
    """
    if not isinstance(text, str): return ""
    
    # LAYER 1: Google API (Requires Internet)
    if USE_GOOGLE:
        try:
            return clean_and_process(text)
        except Exception:
            # Internet failed? Fall through to Layer 2
            pass 
            
    # LAYER 2: Offline Converter (No Internet needed)
    if USE_OFFLINE:
        try:
            return offline_converter.process_sentence(text)
        except Exception:
            pass
            
    # LAYER 3: Basic Regex Fallback
    text = re.sub(r'http\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ==========================================
# 3. LOAD MODEL
# ==========================================
print("🔄 Loading SinhSafe Model...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.to(device)
    model.eval()
    print(f"✅ Model Loaded on {device}")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Model failed to load. {e}")
    exit()

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
def get_predictions_batch(texts, batch_size=16):
    all_labels = []
    all_confs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        try:
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding="max_length", 
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)

            conf_scores, pred_indices = torch.max(probs, dim=1)

            for idx, conf in zip(pred_indices, conf_scores):
                all_labels.append(model.config.id2label[idx.item()])
                all_confs.append(f"{conf.item():.0%}")
        except Exception:
            for _ in batch_texts:
                all_labels.append("Error")
                all_confs.append("0%")
    return all_labels, all_confs

# ==========================================
# 5. MAIN EXECUTION LOOP
# ==========================================
combined_data = []

print("\n🚀 Starting Robust Processing Job...")

for file_path in FILE_LIST:
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} not found. Skipping.")
        continue
        
    print(f"\n📂 Processing {file_path}...")
    try:
        df = pd.read_excel(file_path)
        
        # Smart Column Detection
        if 'comment' in df.columns:
            text_col = 'comment'
        elif 'text' in df.columns:
            text_col = 'text'
        else:
            text_col = df.columns[0]
            
        print(f"   ℹ️ Target Column: '{text_col}'")

        # 1. CLEANING (Hybrid Mode)
        print("   ⏳ Cleaning (Google → Offline Fallback)...")
        clean_texts = []
        for text in tqdm(df[text_col], desc="Cleaning"):
            clean_texts.append(smart_clean_text(text))
        
        # 2. PREDICTION
        print("   🧠 Running Predictions...")
        labels, confidences = get_predictions_batch(clean_texts)
        
        # 3. STORE
        df['cleaned_text'] = clean_texts
        df['label'] = labels
        df['prediction_confidence'] = confidences
        
        combined_data.append(df)
        print(f"   ✅ Finished {file_path}")
        
    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")

# ==========================================
# 6. SAVE OUTPUTS
# ==========================================
if combined_data:
    print("\n💾 Saving Final Files...")
    master_df = pd.concat(combined_data, ignore_index=True)

    for label_name, filename in OUTPUT_FILES.items():
        subset = master_df[master_df['label'] == label_name]
        if not subset.empty:
            subset.to_excel(filename, index=False)
            print(f"   ✅ Saved {len(subset)} rows to '{filename}'")
    print("\n🎉 All Done! Good Morning!")
else:
    print("\n❌ No data processed.")