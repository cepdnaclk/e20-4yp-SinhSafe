import pandas as pd
import requests
import re
import time
import os
import nltk
from nltk.corpus import words

# --- Setup ---
# Download English words if not present
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# 1. Configuration
INPUT_DIR = r"data/raw"   # Your folder path from the image
OUTPUT_DIR = r"data/processed_ground_truth"
FILES = ['cyberbullying.xlsx', 'normal.xlsx', 'offensive.xlsx']
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. English Vocabulary (for filtering)
english_vocab = set(words.words())
# Add common SL-context English words to keep
english_vocab.update(['phone', 'gym', 'car', 'bus', 'laptop', 'internet', 'online', 'ok', 'set', 'shape'])

# 3. Cache to save API calls (Critical for 5000 rows)
transliteration_cache = {}

def is_sinhala(text):
    """Checks if a word is already Sinhala (Unicode)"""
    return bool(re.search('[\u0D80-\u0DFF]', text))

def remove_usernames(text):
    """Removes @mentions like @USER, @kamal from text."""
    # Ensure text is string
    text = str(text)
    # Regex to remove @ followed by word characters
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def google_transliterate(word):
    """
    Uses Google Input Tools API. 
    If it fails or finds no Sinhala match, it returns the ORIGINAL English word.
    """
    # 1. Check cache first
    if word in transliteration_cache:
        return transliteration_cache[word]

    try:
        url = "https://inputtools.google.com/request?text=" + word + "&itc=si-t-i0-und&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8"
        # Added timeout so it doesn't get stuck forever
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            res_json = response.json()
            
            # 2. Check if we actually got a suggestion
            if res_json[0] == 'SUCCESS' and len(res_json) > 1 and len(res_json[1]) > 0:
                sinhala_word = res_json[1][0][1][0]
                transliteration_cache[word] = sinhala_word
                return sinhala_word
            else:
                # API returned Success but list was empty (The "Antonio" case)
                # print(f" - No Sinhala match for '{word}', keeping original.") 
                return word 

    except Exception as e:
        # 3. If Connection fails or any other error, just ignore and keep text
        # print(f" - API Error for '{word}', keeping original.")
        return word
    
    # Final fallback
    return word

def clean_and_process(text):
    if not isinstance(text, str):
        return ""
    
    # --- NEW: CLEAN NOISE FIRST ---
    # We remove @USER before splitting, so we don't accidentally 
    # try to transliterate the username.
    text = remove_usernames(text)
    # ------------------------------

    tokens = text.split()
    processed_tokens = []
    
    for token in tokens:
        # Strip punctuation for checking
        clean_token = re.sub(r'[^\w]', '', token).lower()
        
        # LOGIC 1: Already Sinhala -> Keep
        if is_sinhala(token):
            processed_tokens.append(token)
            
        # LOGIC 2: Numbers/Symbols -> Keep
        elif not clean_token.isalpha():
            processed_tokens.append(token)
            
        # LOGIC 3: Real English Word -> Keep
        elif len(clean_token) > 2 and clean_token in english_vocab:
            processed_tokens.append(token)
            
        # LOGIC 4: Singlish -> Google API
        else:
            # Only hit API if we have a valid word
            if clean_token:
                converted = google_transliterate(clean_token)
                processed_tokens.append(converted)
            else:
                processed_tokens.append(token)

    return " ".join(processed_tokens)

# --- Main Execution ---
if __name__ == "__main__":
    for file_name in FILES:
        input_path = os.path.join(INPUT_DIR, file_name)
        
        if os.path.exists(input_path):
            print(f"Processing {file_name} with Google API...")
            
            # Load Excel
            df = pd.read_excel(input_path)
            
            # Ensure 'comment' column exists (Adjust column name if needed)
            # Assuming the text column is the first one or named 'text'/'comment'
            # Let's auto-detect text column if not sure, or assume 'comment'
            text_col = 'comment' if 'comment' in df.columns else df.columns[0]
            print(f"  - Using column: {text_col}")

            # Apply Processing (with a progress counter logic if needed)
            # We use a simple loop to handle API rate limiting better if needed
            results = []
            total = len(df)
            for i, row in df.iterrows():
                if i % 100 == 0: print(f"    Row {i}/{total}")
                results.append(clean_and_process(row[text_col]))
                
            df['cleaned_text'] = results
            
            # Save
            output_path = os.path.join(OUTPUT_DIR, f"processed_{file_name}")
            df.to_excel(output_path, index=False)
            print(f"  - Saved to {output_path}")
        else:
            print(f"File not found: {input_path}")
            
    print("Done! Ground truth processing complete.")