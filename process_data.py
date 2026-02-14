import pandas as pd
import sys
import os
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
current_dir = os.getcwd()

# Input: Where your manually labeled consolidated files are
INPUT_DIR = os.path.join(current_dir, "data", "raw")

# Output: Where the cleaned, tokenized files will be saved
OUTPUT_DIR = os.path.join(current_dir, "data", "processed_ground_truth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The specific files to process
FILE_LIST = [
    'consolidated_harassment.xlsx',
    'consolidated_normal.xlsx',
    'consolidated_offensive.xlsx'
]

# ==========================================
# 2. SETUP CLEANING ENGINE
# ==========================================
# We add 'src' to the system path just in case your script is there
sys.path.append(os.path.join(current_dir, 'src'))

try:
    # Attempt to import your cleaning logic
    from process_ground_truth import clean_and_process
    print("✅ Loaded: process_ground_truth (Google API / Transliteration Enabled)")
except ImportError:
    # If the file is in the root directory instead of src
    try:
        import process_ground_truth
        from process_ground_truth import clean_and_process
        print("✅ Loaded: process_ground_truth (from root)")
    except ImportError:
        print("❌ CRITICAL ERROR: Could not find 'process_ground_truth.py'")
        print("   Make sure the file exists in your folder.")
        sys.exit(1)

# ==========================================
# 3. PROCESSING LOOP
# ==========================================
def process_ground_truth_files():
    print(f"\n🚀 Starting Ground Truth Processing...")
    print(f"📂 Reading from: {INPUT_DIR}")
    print(f"💾 Saving to:   {OUTPUT_DIR}\n")

    for filename in FILE_LIST:
        input_path = os.path.join(INPUT_DIR, filename)
        
        if not os.path.exists(input_path):
            print(f"⚠️  Warning: {filename} not found in data/raw. Skipping.")
            continue

        print(f"Processing {filename}...")
        
        try:
            # 1. Load Data
            df = pd.read_excel(input_path)
            
            # 2. Detect Columns (Case Insensitive)
            cols = {c.lower(): c for c in df.columns}
            
            # Find Text Column
            if 'comment' in cols:
                text_col = cols['comment']
            elif 'text' in cols:
                text_col = cols['text']
            else:
                text_col = df.columns[0] # Fallback to first column

            # Find Label Column
            if 'label' in cols:
                label_col = cols['label']
            else:
                label_col = None
                print("   ⚠️  Warning: No 'label' column found. Checking filename...")

            print(f"   ℹ️  Using text column: '{text_col}'")

            # 3. Clean & Tokenize Text
            # This runs your 'clean_and_process' function (Google API) on every row
            cleaned_texts = []
            for text in tqdm(df[text_col], desc="   Cleaning & Tokenizing"):
                cleaned_texts.append(clean_and_process(text))
            
            df['cleaned_text'] = cleaned_texts

            # 4. Process Labels (Lowercase Only)
            if label_col:
                # Force lowercase and strip spaces
                df[label_col] = df[label_col].astype(str).str.lower().str.strip()
            else:
                # If no label col exists, create one based on filename
                if 'harassment' in filename:
                    df['label'] = 'harassment'
                elif 'offensive' in filename:
                    df['label'] = 'offensive'
                elif 'normal' in filename:
                    df['label'] = 'normal'

            # 5. Save Final File
            # We add a 'processed_' prefix to keep things organized
            output_filename = f"processed_{filename}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Keep only the essential columns
            # If label_col was found, use it; otherwise use 'label' created above
            final_label_col = label_col if label_col else 'label'
            final_df = df[[text_col, final_label_col, 'cleaned_text']]
            
            # Rename columns to standard names for training
            final_df.columns = ['comment', 'label', 'cleaned_text']

            # Remove empty rows if cleaning failed completely
            final_df = final_df.dropna(subset=['cleaned_text', 'label'])
            
            final_df.to_excel(output_path, index=False)
            print(f"   ✅ Saved {len(final_df)} rows to {output_filename}\n")

        except Exception as e:
            print(f"   ❌ Failed to process {filename}: {e}\n")

    print("="*30)
    print("🎉 All Ground Truth Data Processed & Ready for Training!")
    print("="*30)

if __name__ == "__main__":
    process_ground_truth_files()