import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import PeftModel
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    EarlyStoppingCallback  # <--- ADD THIS
)

# --- 1. NEW V2 CONFIGURATION ---
BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B"  
ADAPTER_ID = "./SinLlama_Local"                 
TOKENIZER_ID = "polyglots/Extended-Sinhala-LLaMA"

# Changing directories so we DO NOT overwrite your first model
OUTPUT_DIR = "./results/sinhsafe_v3"
FINAL_MODEL_DIR = "./models/sinhsafe_v3_regularized"

# --- 2. LOAD TOKENIZER & BASE MODEL ---
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True 
)

model.resize_token_embeddings(len(tokenizer))

# --- 3. LOAD THE SINLLAMA ADAPTER ---
model = PeftModel.from_pretrained(model, ADAPTER_ID, is_trainable=True)

# --- 4. DATA PREPARATION (THE 80/10/10 SPLIT) ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Categorize the following Sinhala/Singlish text into exactly one of three categories: Normal, Offensive, or Harassment. 
Use the following strict definitions to make your decision:
- Harassment: Targeted behavior meant to degrade or intimidate. Includes threats of violence, encouraging self-harm, or attacking someone's family, women, religion, and ethnicity-based attacks.
- Offensive: Content that violates social norms or decorum. General vulgarity, crude jokes, or "blue" humor without a specific target.
- Normal: Standard, respectful communication. Professional, casual, or friendly dialogue that follows social etiquette.

### Input:
{}

### Response:
{}"""

def format_dataset(df):
    formatted_texts = []
    for _, row in df.iterrows():
        text = alpaca_prompt.format(row['cleaned_text'], row['label']) + tokenizer.eos_token
        formatted_texts.append(text)
    return Dataset.from_pandas(pd.DataFrame({'text': formatted_texts}))

def load_and_split_data():
    df_harass = pd.read_excel("data/processed_ground_truth/processed_consolidated_harassment.xlsx")
    df_offen = pd.read_excel("data/processed_ground_truth/processed_consolidated_offensive.xlsx")
    df_norm = pd.read_excel("data/processed_ground_truth/processed_consolidated_normal.xlsx")
    
    df_norm['label'] = "Normal"
    df_offen['label'] = "Offensive"
    df_harass['label'] = "Harassment"
    
    # Combine all into one massive dataframe
    df_all = pd.concat([df_harass, df_offen, df_norm])
    
    # --- STRICT SPLITTING ---
    # 1. Split off 20% for Val/Test combined (80% stays in Train)
    train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['label'])
    
    # 2. Split that 20% directly in half (10% Val, 10% Test)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # --- BALANCE ONLY THE TRAINING SET ---
    # We must not balance Val or Test to keep them true to the real world
    max_size = train_df['label'].value_counts().max()
    
    train_bal = pd.concat([
        train_df[train_df['label'] == 'Harassment'].sample(max_size, replace=True, random_state=42),
        train_df[train_df['label'] == 'Offensive'].sample(max_size, replace=True, random_state=42),
        train_df[train_df['label'] == 'Normal'].sample(max_size, replace=True, random_state=42)
    ]).sample(frac=1, random_state=42)
    
    # Save the Test set to a CSV so you can evaluate it later!
    test_df.to_csv("sinhsafe_test_set.csv", index=False)
    print(f"📁 Saved {len(test_df)} unseen test rows to 'sinhsafe_test_set.csv'")

    return format_dataset(train_bal), format_dataset(val_df)

# Load the newly split datasets
train_dataset, val_dataset = load_and_split_data()

# --- 5. TRAINING LOOP (WITH EARLY STOPPING) ---
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, 
    args=SFTConfig(
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=2,       
        learning_rate=5e-5,       
        weight_decay=0.05,        
        bf16=True, 
        logging_steps=10,
        
        # --- EARLY STOPPING SETTINGS ---
        eval_strategy="steps",    
        eval_steps=50,            
        save_strategy="steps",        # MUST match eval_strategy
        save_steps=50,                # MUST match eval_steps
        save_total_limit=3,
        load_best_model_at_end=True,  # Rewinds to the lowest eval_loss!
        metric_for_best_model="eval_loss",
        greater_is_better=False,      # Because lower loss is better
        
        output_dir=OUTPUT_DIR,
        report_to="none"
    ),
    # Add the callback here! patience=3 means it will wait for 3 bad evaluations (150 steps) before pulling the plug.
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
)

print("🚀 Starting SinhSafe V2 Training with Validation on RTX 3090 Ti...")
trainer.train()

trainer.model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"✅ V3 Training complete! New model safely saved to {FINAL_MODEL_DIR}")