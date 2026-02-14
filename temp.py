import pandas as pd
import os

# --- Configuration ---
file_1 = "cyberbullying_model_predict.xlsx"
file_2 = "cyberbullying_model_predict_2.xlsx"
output_file = "manual_review_batch.xlsx"

# WE WANT THE "CONFUSED" DATA
LOWER_LIMIT = 0.40  # 40%
UPPER_LIMIT = 0.85  # 85%

def clean_percentage(val):
    if pd.isna(val): return None
    s_val = str(val).strip()
    if '%' in s_val:
        try: return float(s_val.replace('%', '')) / 100.0
        except: return None
    try: return float(s_val)
    except: return None

def filter_confusion_zone():
    print("Loading files...")
    df1 = pd.read_excel(file_1) if os.path.exists(file_1) else pd.DataFrame()
    df2 = pd.read_excel(file_2) if os.path.exists(file_2) else pd.DataFrame()
    df_all = pd.concat([df1, df2], ignore_index=True)

    # Find column
    target_col = None
    for col in df_all.columns:
        if "conf" in str(col).lower() or "pred" in str(col).lower():
            target_col = col
            break
            
    if not target_col:
        print("❌ Column not found!")
        return

    # Clean Data
    df_all['clean_score'] = df_all[target_col].apply(clean_percentage)
    df_clean = df_all.dropna(subset=['clean_score'])

    # FILTER: BETWEEN 40% AND 85%
    print(f"Filtering for Confusion Zone: {LOWER_LIMIT} to {UPPER_LIMIT}...")
    
    # Check scaling (0-1 vs 0-100)
    if df_clean['clean_score'].max() > 1.0:
        low = LOWER_LIMIT * 100
        high = UPPER_LIMIT * 100
    else:
        low = LOWER_LIMIT
        high = UPPER_LIMIT

    # Logic: Score is GREATER than Low AND LESS than High
    mask = (df_clean['clean_score'] >= low) & (df_clean['clean_score'] <= high)
    df_filtered = df_clean[mask].copy()
    
    # Sort by score (descending) so you see the "almost sure" ones first
    df_filtered = df_filtered.sort_values(by='clean_score', ascending=False)
    
    # Save
    df_filtered.to_excel(output_file, index=False)
    print(f"🎉 Saved {len(df_filtered)} 'Hard' examples to {output_file}")
    print("Go label these manually to make your model smarter!")

if __name__ == "__main__":
    filter_confusion_zone()