import pandas as pd
import os

# --- Configuration ---
# Based on your screenshots for the data/labelled_data folder
DATA_DIR = "data/labelled_data"
OUTPUT_FILE = "sinhsafe_gold_standard_unanimous.csv"

def generate_gold_standard():
    print("🤖 Initializing Ensemble Megazord...")

    # 1. Load the three labeled files
    # Note: Using your file names from the screenshot
    try:
        df_xlm = pd.read_csv(os.path.join(DATA_DIR, "labelled_xlm.csv"))
        df_sinbert = pd.read_csv(os.path.join(DATA_DIR, "labelled_sinbert.csv"))
        df_sinllama = pd.read_csv(os.path.join(DATA_DIR, "labelled_sinllama.csv"))
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find files. Check paths. {e}")
        return

    print(f"📊 Loaded: XLM ({len(df_xlm)}), SinBERT ({len(df_sinbert)}), SinLlama ({len(df_sinllama)})")

    # 2. Rename columns to avoid collisions before merging
    # We keep 'cleaned_text' as the common key
    df_xlm = df_xlm.rename(columns={'xlm_label': 'label_xlm'})
    df_sinbert = df_sinbert.rename(columns={'sinbert_label': 'label_sinbert'})
    df_sinllama = df_sinllama.rename(columns={'sinllama_label': 'label_sinllama'})

    # 3. Merge dataframes on 'cleaned_text'
    # We use inner join to ensure only text existing in ALL three is processed
    merged_df = df_xlm.merge(df_sinbert, on='cleaned_text', how='inner')
    merged_df = merged_df.merge(df_sinllama, on='cleaned_text', how='inner')

    print(f"🔗 Merged data. Total overlapping comments: {len(merged_df)}")

    # 4. FILTER: The Unanimous Agreement Rule
    # Keep only where all three labels match perfectly
    gold_df = merged_df[
        (merged_df['label_xlm'] == merged_df['label_sinbert']) & 
        (merged_df['label_sinbert'] == merged_df['label_sinllama'])
    ].copy()

    # 5. Final Formatting
    # Set the 'universal_label' from any of the matching columns
    gold_df['label'] = gold_df['label_xlm']

    # Select and order columns as requested:
    # comment, cleaned_text, xlm_accuracy, sinbert_accuracy, sinllama_accuracy, label
    # Note: 'comment' is the original raw text, 'cleaned_text' is the processed version
    final_columns = [
        'comment', 
        'cleaned_text', 
        'xlm_accuracy', 
        'sinbert_accuracy', 
        'sinllama_accuracy', 
        'label'
    ]
    
    # Check if 'comment' exists (if not, we use cleaned_text for both)
    if 'comment' not in gold_df.columns:
        gold_df['comment'] = gold_df['cleaned_text']

    gold_standard = gold_df[final_columns]

    # 6. Save and Report
    gold_standard.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"🏆 GOLD STANDARD CREATED")
    print(f"✅ Total rows with 3-way agreement: {len(gold_standard)}")
    print(f"📈 Agreement Rate: {(len(gold_standard)/len(merged_df))*100:.2f}%")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print("="*40)

    # Class breakdown in the final set
    print("\n📝 Final Class Distribution:")
    print(gold_standard['label'].value_counts())

if __name__ == "__main__":
    generate_gold_standard()