import pandas as pd
import os

DATA_DIR = "data/labelled_data"
# The range we want to manually verify
LOWER_BOUND = 0.80
UPPER_BOUND = 0.8999
OUTPUT_FILE = "harassment_to_verify_80_90.csv"

def extract_for_verification():
    print(f"⛏️ Mining for 'Borderline' Harassment ({LOWER_BOUND*100}% - {UPPER_BOUND*100}%)...")

    # 1. Load the three labeled files
    df_xlm = pd.read_csv(os.path.join(DATA_DIR, "labelled_xlm.csv"))
    df_sinbert = pd.read_csv(os.path.join(DATA_DIR, "labelled_sinbert.csv"))
    df_sinllama = pd.read_csv(os.path.join(DATA_DIR, "labelled_sinllama.csv"))

    # 2. Extract rows in the 80-90% range for Harassment
    h_xlm = df_xlm[(df_xlm['xlm_label'] == 'Harassment') & (df_xlm['xlm_accuracy'] >= LOWER_BOUND) & (df_xlm['xlm_accuracy'] <= UPPER_BOUND)].copy()
    h_bert = df_sinbert[(df_sinbert['sinbert_label'] == 'Harassment') & (df_sinbert['sinbert_accuracy'] >= LOWER_BOUND) & (df_sinbert['sinbert_accuracy'] <= UPPER_BOUND)].copy()
    h_llama = df_sinllama[(df_sinllama['sinllama_label'] == 'Harassment') & (df_sinllama['sinllama_accuracy'] >= LOWER_BOUND) & (df_sinllama['sinllama_accuracy'] <= UPPER_BOUND)].copy()

    # 3. Standardize and combine
    h_xlm = h_xlm.rename(columns={'xlm_label': 'label', 'xlm_accuracy': 'confidence'})
    h_bert = h_bert.rename(columns={'sinbert_label': 'label', 'sinbert_accuracy': 'confidence'})
    h_llama = h_llama.rename(columns={'sinllama_label': 'label', 'sinllama_accuracy': 'confidence'})

    combined = pd.concat([h_xlm, h_bert, h_llama], ignore_index=True)

    # 4. Deduplicate so you don't check the same sentence twice
    # We sort by confidence so you see the "most likely" ones first
    combined = combined.sort_values(by='confidence', ascending=False)
    final_to_check = combined.drop_duplicates(subset=['cleaned_text'])

    # 5. Save for your manual review
    # Keep it simple: you just need the text to read it
    final_to_check[['comment', 'cleaned_text', 'confidence']].to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"✅ Created verification file: {OUTPUT_FILE}")
    print(f"📝 Total rows to check: {len(final_to_check)}")
    print("="*40)
    print("💡 Tip: Open this in Excel, read the 'comment', and delete the rows that aren't actually harassment.")

if __name__ == "__main__":
    extract_for_verification()