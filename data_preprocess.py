import pandas as pd
import os

# Import the exact function from your existing script
from src.process_ground_truth import clean_and_process

# File paths
INPUT_FILE = "data/unlabelled_data/BigData.xlsx"
OUTPUT_FILE = "data/processed_unlabelled_data/cleaned_BigData.csv"

def process_big_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Cannot find {INPUT_FILE}")
        return

    print(f"🚀 Loading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    
    print(f"⚙️ Processing {len(df)} rows using existing clean_and_process function...")
    
    # Pass the data to your existing function, handling empty cells
    df['cleaned_text'] = df['comment'].apply(lambda x: clean_and_process(str(x)) if pd.notna(x) else "")
    
    # Filter to only the columns you want
    final_df = df[['comment', 'cleaned_text', 'label']]
    
    # Save as CSV
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"✅ Done! Saved clean data to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_big_data()