import pandas as pd
import os

# --- Configuration ---
DATA_DIR = "data/labelled_data"
OUTPUT_FILE = "sinhsafe_xlm_sinbert_consensus.csv"

def generate_dual_consensus():
    print("⚔️ Initializing Dual-Model Ensemble (XLM + SinBERT)...")

    # 1. Load the files
    try:
        df_xlm = pd.read_csv(os.path.join(DATA_DIR, "labelled_xlm.csv"))
        df_sinbert = pd.read_csv(os.path.join(DATA_DIR, "labelled_sinbert.csv"))
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    print(f"📊 Loaded: XLM ({len(df_xlm)}), SinBERT ({len(df_sinbert)})")

    # 2. Rename for clarity
    df_xlm = df_xlm.rename(columns={'xlm_label': 'label_xlm'})
    df_sinbert = df_sinbert.rename(columns={'sinbert_label': 'label_sinbert'})

    # 3. Inner Join on text
    merged_df = df_xlm.merge(df_sinbert, on='cleaned_text', how='inner')

    # 4. FILTER: The 2-Way Agreement Rule
    # Keep only where XLM and SinBERT agree perfectly
    consensus_df = merged_df[merged_df['label_xlm'] == merged_df['label_sinbert']].copy()

    # 5. Final Formatting
    consensus_df['label'] = consensus_df['label_xlm']

    # Columns: comment, cleaned_text, xlm_accuracy, sinbert_accuracy, label
    # (Leaving sinllama out of the final selection as requested)
    final_columns = [
        'comment', 
        'cleaned_text', 
        'xlm_accuracy', 
        'sinbert_accuracy', 
        'label'
    ]
    
    if 'comment' not in consensus_df.columns:
        consensus_df['comment'] = consensus_df['cleaned_text']

    final_df = consensus_df[final_columns]

    # 6. Save
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"🏆 DUAL CONSENSUS CREATED")
    print(f"✅ Total rows with 2-way agreement: {len(final_df)}")
    print(f"📈 Match Rate: {(len(final_df)/len(merged_df))*100:.2f}%")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print("="*40)

    print("\n📝 New Class Distribution:")
    print(final_df['label'].value_counts())

if __name__ == "__main__":
    generate_dual_consensus()