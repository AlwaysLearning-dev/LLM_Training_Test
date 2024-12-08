from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Paths
base_model_path = '/home/bob/llama/'  # Base model directory
lora_model_path = './fine-tuned-llama-noquant'  # LoRA fine-tuned weights

# Load the base model
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")

# Load the fine-tuned LoRA adapters
model = PeftModel.from_pretrained(model, lora_model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set

print("Fine-tuned LLM loaded. Type your input and press Enter to generate text.")
print("Type 'exit' to quit.")

# Start interactive loop
while True:
    # Prompt user for input
    input_text = input("\nYour input: ")

    # Exit condition
    if input_text.lower() == "exit":
        print("Exiting. Goodbye!")
        break

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)

    # Generate output
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        max_length=512,              # Allow up to 512 tokens
        temperature=0.7,             # Adjust creativity
        num_beams=5,                 # Improve output quality
        do_sample=True,              # Enable variability
        pad_token_id=tokenizer.eos_token_id,  # Explicitly set padding token
        eos_token_id=tokenizer.eos_token_id,  # Ensure proper stopping
        no_repeat_ngram_size=3,      # Prevent repetition
        early_stopping=True          # Stop at <eos> if reached
    )

    # Decode and print output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated output:")
    print(decoded_output)