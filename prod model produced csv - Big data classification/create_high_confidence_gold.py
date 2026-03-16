import pandas as pd
import os

# --- Configuration ---
DATA_DIR = "data/labelled_data"
THRESHOLD = 0.90
OUTPUT_FILE = "sinhsafe_high_confidence_union.csv"

def generate_expert_union():
    print(f"🎯 Initializing Expert Extraction (Threshold: {THRESHOLD*100}%)...")

    # 1. Load the three labeled files
    try:
        df_xlm = pd.read_csv(os.path.join(DATA_DIR, "labelled_xlm.csv"))
        df_sinbert = pd.read_csv(os.path.join(DATA_DIR, "labelled_sinbert.csv"))
        df_sinllama = pd.read_csv(os.path.join(DATA_DIR, "labelled_sinllama.csv"))
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 2. Add model source identifier and standardize columns
    df_xlm['source'] = 'xlm'
    df_xlm = df_xlm.rename(columns={'xlm_label': 'label', 'xlm_accuracy': 'confidence'})
    
    df_sinbert['source'] = 'sinbert'
    df_sinbert = df_sinbert.rename(columns={'sinbert_label': 'label', 'sinbert_accuracy': 'confidence'})
    
    df_sinllama['source'] = 'sinllama'
    df_sinllama = df_sinllama.rename(columns={'sinllama_label': 'label', 'sinllama_accuracy': 'confidence'})

    # 3. Combine them all into one massive list (3 x 148k rows)
    combined_all = pd.concat([df_xlm, df_sinbert, df_sinllama], ignore_index=True)

    # 4. Filter for only those that hit the 90% mark
    high_conf_pool = combined_all[combined_all['confidence'] >= THRESHOLD].copy()
    
    print(f"🔍 Found {len(high_conf_pool)} total high-confidence predictions across all files.")

    # 5. DEDUPLICATION (Crucial Step)
    # If multiple models were >90% sure about the same sentence, we keep the one 
    # with the absolute highest confidence.
    # We sort by cleaned_text and confidence (descending)
    high_conf_pool = high_conf_pool.sort_values(by=['cleaned_text', 'confidence'], ascending=[True, False])
    
    # Drop duplicates so each 'cleaned_text' appears only once
    final_dataset = high_conf_pool.drop_duplicates(subset=['cleaned_text'], keep='first')

    # 6. Save
    # Keep requested columns: comment, cleaned_text, label, confidence, source
    final_dataset.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"🏆 HIGH CONFIDENCE DATASET CREATED")
    print(f"✅ Unique rows with at least one model >90%: {len(final_dataset)}")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print("="*40)

    print("\n📝 Distribution of Labels (Expert Union):")
    print(final_dataset['label'].value_counts())

if __name__ == "__main__":
    generate_expert_union()