import pandas as pd
import os

# --- Configuration ---
DATA_DIR = "data/labelled_data"
UPPER_BOUND = 0.80
OUTPUT_FILE = "hidden_harassment_consensus_under_80.csv"

def mine_dual_hidden_harassment():
    print(f"🕵️‍♂️ Mining for Hidden Harassment (XLM + SinBERT agreement <{UPPER_BOUND*100}%)...")

    # 1. Load the two specialized models
    try:
        df_xlm = pd.read_csv(os.path.join(DATA_DIR, "labelled_xlm.csv")).rename(
            columns={'xlm_label': 'l_xlm', 'xlm_accuracy': 'c_xlm'})
        df_bert = pd.read_csv(os.path.join(DATA_DIR, "labelled_sinbert.csv")).rename(
            columns={'sinbert_label': 'l_bert', 'sinbert_accuracy': 'c_bert'})
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 2. Merge on cleaned_text
    merged = df_xlm.merge(df_bert, on='cleaned_text')

    # 3. FILTER LOGIC:
    # - Both XLM and SinBERT must say Harassment
    # - We target cases where they are 'unsure' (under 80%) 
    #   to find the tricky ones for your manual check.
    hidden_h = merged[
        (merged['l_xlm'] == 'Harassment') & 
        (merged['l_bert'] == 'Harassment') & 
        (merged['c_xlm'] < UPPER_BOUND)
    ].copy()

    # 4. Cleanup for manual review
    final_cols = ['comment', 'cleaned_text', 'l_xlm', 'c_xlm', 'l_bert', 'c_bert']
    
    # Ensure 'comment' exists
    if 'comment' not in hidden_h.columns:
        hidden_h['comment'] = hidden_h['cleaned_text']
        
    result_df = hidden_h[final_cols].drop_duplicates(subset=['cleaned_text'])

    # 5. Save
    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"🏆 DUAL-MODEL HARASSMENT MINE COMPLETE")
    print(f"✅ Found {len(result_df)} rows where XLM & SinBERT agree (under 80% confidence).")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print("="*40)
    print("💡 Manual Step: Check these and keep only the true Harassment rows to hit your 5k goal.")

if __name__ == "__main__":
    mine_dual_hidden_harassment()