import torch
from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType

# --- CONFIGURATION ---
MODEL_ID = "keshan/Sinhala-Llama-2-7b-chat" # Or the specific SinLlama path you have
OUTPUT_DIR = "./results/sinllama_finetuned"

# 1. Load Data (Same logic as your XLM-R script)
def load_data():
    # Load your excel files here like before
    df_harass = pd.read_excel("data/processed_ground_truth/processed_consolidated_harassment.xlsx")
    df_offen = pd.read_excel("data/processed_ground_truth/processed_consolidated_offensive.xlsx")
    df_norm = pd.read_excel("data/processed_ground_truth/processed_consolidated_normal.xlsx")
    
    # Labels: Normal=0, Offensive=1, Harassment=2
    df_norm['label'] = 0
    df_offen['label'] = 1
    df_harass['label'] = 2
    
    df = pd.concat([df_harass, df_offen, df_norm]).sample(frac=1, random_state=42)
    df = df[['cleaned_text', 'label']].dropna().rename(columns={'cleaned_text': 'text'})
    df['text'] = df['text'].astype(str)
    
    # Simple split
    train_size = int(0.8 * len(df))
    train_dataset = Dataset.from_pandas(df[:train_size])
    eval_dataset = Dataset.from_pandas(df[train_size:])
    
    return DatasetDict({"train": train_dataset, "test": eval_dataset})

dataset = load_data()

# 2. Quantization Config (The Magic for RTX 3090)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Loads model in 4-bit to save VRAM
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# 3. Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token # Fix for Llama

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=3, # Normal, Offensive, Harassment
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id

# 4. Apply LoRA (The "Fine-Tuning" part)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,           # Rank
    lora_alpha=32,  # Scaling
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"] # Target attention layers
)
model = get_peft_model(model, peft_config)
print(" trainable params:")
model.print_trainable_parameters()

# 5. Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-4, # Higher LR for LoRA
    per_device_train_batch_size=4, # Keep small for VRAM
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True, # Enable mixed precision
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

print("🚀 Starting SinLlama Training...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
print("✅ Saved SinLlama Adapter!")
