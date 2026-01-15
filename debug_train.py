import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# 1. Create Fake Data (10 rows only)
print("Creating fake data...")
df = pd.DataFrame({
    'text': ["This is a test sentence " * 10] * 20,
    'label': [0, 1] * 10
})

# 2. Tokenize
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
dataset = Dataset.from_pandas(df)
def tokenize(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=64)
dataset = dataset.map(tokenize, batched=True)

# 3. Load Model
print("Loading Model...")
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=2).to("cuda")

# 4. Train with MINIMUM settings
training_args = TrainingArguments(
    output_dir="./debug_results",
    num_train_epochs=1,
    per_device_train_batch_size=2,  # Tiny batch
    dataloader_num_workers=0,       # No parallelism
    fp16=True,                      # Fast mode
    report_to="none"
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=dataset
)

print("Starting Fake Training...")
trainer.train()
print("SUCCESS! Training finished.")