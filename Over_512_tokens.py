from transformers import AutoTokenizer
import json

# Load dataset
file_path = 'rules_dataset.jsonl'
data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('/home/bob/llama/', use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Calculate token lengths
token_lengths = [
    len(tokenizer(example['instruction'] + example['output'])['input_ids'])
    for example in data
]

# Count elements exceeding 512 tokens
over_1024_count = sum(1 for length in token_lengths if length > 1024)

print(f"Number of elements over 1024 tokens: {over_1024_count}")