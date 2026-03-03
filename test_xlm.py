import torch
import torch.nn.functional as F
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import sys
import os

# ==========================================
# 1. SETUP PATHS & IMPORTS
# ==========================================
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')

# Add 'src' to system path so Python can find process_ground_truth.py
if src_path not in sys.path:
    sys.path.append(src_path)

# Import your cleaning function
try:
    from process_ground_truth import clean_and_process
    print("✅ Loaded preprocessing logic from src/")
except ImportError as e:
    print(f"⚠️ Warning: Could not import 'clean_and_process' from src. ({e})")
    print("   Using fallback (no cleaning).")
    def clean_and_process(text): return text

# ==========================================
# 2. MODEL CONFIGURATION
# ==========================================
# Force absolute path to avoid confusion
MODEL_PATH = os.path.join(current_dir, "models", "best_cv_model")

print("🔄 Loading SinhSafe Model...")

# Safe Device Selection (Prevents GPU Freeze)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("ℹ️ Using CPU")

# Load Model
try:
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Make sure you ran 'pip install safetensors'")
    exit()

model.to(device)
model.eval()

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
def predict_text(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128, 
        padding="max_length"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    pred_idx = torch.argmax(probs).item()
    confidence = probs[0][pred_idx].item()
    predicted_label = model.config.id2label[pred_idx]
    
    return predicted_label, confidence

# ==========================================
# 4. MAIN LOOP
# ==========================================
print("\n" + "="*50)
print("     🤖 SINHSAFE LIVE DEMO    ")
print("="*50)
print("Type a Singlish/Sinhala sentence and press Enter.")

while True:
    user_input = input("\n📝 Enter Text: ")
    
    if user_input.lower() in ['exit', 'q']:
        break
    if not user_input.strip():
        continue

    # --- Step 1: Clean/Transliterate ---
    print("   ⏳ Processing...", end="\r")
    cleaned_input = clean_and_process(user_input)
    
    # --- Step 2: Predict ---
    label, conf = predict_text(cleaned_input)

    # Formatting
    color = "\033[92m" if label == "Normal" else "\033[91m"
    reset = "\033[0m"
    
    print(" " * 20, end="\r") # Clear loading text
    print(f"   🔍 Input: '{cleaned_input}'")
    print(f"👉 Prediction: {color}{label.upper()}{reset}")
    print(f"📊 Confidence: {conf:.2%}")
    print("-" * 30)