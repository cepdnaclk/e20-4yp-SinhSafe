import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- CONFIGURATION ---
BASE_MODEL_ID = "NousResearch/Meta-Llama-3-8B"
FINETUNED_MODEL_DIR = "./models/sinhsafe_v2_split"# Where your successful weights are saved

print("⏳ Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)

print("⏳ Loading Base Model (Console) in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0}
)

# CRITICAL: Resize the base brain to match the 139k Sinhala tokens you trained on
base_model.resize_token_embeddings(len(tokenizer))

print("⏳ Plugging in SinhSafe Weights (Cartridge)...")
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_DIR)

# The exact same Alpaca prompt format you used during training
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
"""

def predict_category(text):
    # Format the user's text into the Alpaca prompt
    prompt = alpaca_prompt.format(text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate the prediction (do_sample=False forces it to give its most confident answer)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=10, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and extract just the AI's answer
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    category = full_response.split("### Response:\n")[-1].strip()
    return category

print("\n" + "="*50)
print("✅ SinhSafe is LIVE! Type a sentence to test it.")
print("Type 'exit' to quit the program.")
print("="*50 + "\n")

# Interactive Loop
while True:
    user_text = input("✍️  Enter Sinhala/Singlish text: ")
    
    if user_text.lower() in ['exit', 'quit']:
        print("Shutting down SinhSafe...")
        break
        
    if user_text.strip() == "":
        continue
        
    prediction = predict_category(user_text)
    print(f"🤖 SinhSafe Classification: {prediction}\n")