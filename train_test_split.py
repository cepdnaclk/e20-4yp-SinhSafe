import pandas as pd
from sklearn.model_selection import train_test_split

# Load your 6075 consolidated rows
df = pd.concat([pd.read_excel(f"data/processed_ground_truth/processed_consolidated_{cat}.xlsx") for cat in ["harassment", "offensive", "normal"]])

# STRATIFIED SPLIT
train_df, test_df = train_test_split(
    df, 
    test_size=0.10, 
    random_state=42, 
    stratify=df['label']
)

train_df.to_csv("train_90.csv", index=False)
test_df.to_csv("test_10.csv", index=False)
print("✅ Golden splits saved: 90% (train_90.csv) and 10% (test_10.csv)")