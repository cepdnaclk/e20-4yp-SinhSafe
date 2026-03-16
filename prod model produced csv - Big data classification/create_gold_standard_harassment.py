import pandas as pd
import os

# --- Configuration ---
DATA_DIR = "data/labelled_data"
THRESHOLD = 0.90
OUTPUT_FILE = "gold_harassment_90plus.csv"

def generate_gold_harassment():
    print(f"🎯 Extracting Gold Harassment (Threshold: {THRESHOLD*100}%+)...")

    # 1. Load the three labeled files
    try:
        df_xlm = pd.read_csv(os.path.join(DATA_DIR, "labelled_xlm.csv"))
        df_sinbert = pd.read_csv(os.path.join(DATA_DIR, "labelled_sinbert.csv"))
        df_sinllama = pd.read_csv(os.path.join(DATA_DIR, "labelled_sinllama.csv"))
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 2. Standardize columns
    df_xlm = df_xlm.rename(columns={'xlm_label': 'label', 'xlm_accuracy': 'confidence'})
    df_sinbert = df_sinbert.rename(columns={'sinbert_label': 'label', 'sinbert_accuracy': 'confidence'})
    df_sinllama = df_sinllama.rename(columns={'sinllama_label': 'label', 'sinllama_accuracy': 'confidence'})

    # 3. Combine them all
    combined_all = pd.concat([df_xlm, df_sinbert, df_sinllama], ignore_index=True)

    # 4. STRICT FILTER: Only Harassment AND Confidence >= 90%
    gold_h_pool = combined_all[
        (combined_all['label'] == 'Harassment') & 
        (combined_all['confidence'] >= THRESHOLD)
    ].copy()
    
    print(f"🔍 Found {len(gold_h_pool)} high-confidence harassment entries across all models.")

    # 5. DEDUPLICATION
    # Sort by confidence so the best version of each sentence is kept
    gold_h_pool = gold_h_pool.sort_values(by=['cleaned_text', 'confidence'], ascending=[True, False])
    final_h = gold_h_pool.drop_duplicates(subset=['cleaned_text'], keep='first')

    # 6. Save
    final_h.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"🏆 GOLD HARASSMENT FILE CREATED")
    print(f"✅ Total unique Gold Harassment rows: {len(final_h)}")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    generate_gold_harassment()