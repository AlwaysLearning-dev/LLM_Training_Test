from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. Load the dataset
dataset = load_dataset('json', data_files='rules_dataset.jsonl')

# 2. Prepare the data
def combine_fields(example):
    example['text'] = example['instruction'] + example['output']
    return example

dataset = dataset.map(combine_fields)

# 3. Tokenize the data
tokenizer = AutoTokenizer.from_pretrained('/home/bob/llama/', use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=1024, padding='longest')

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. Load the base model (without quantization)
model = AutoModelForCausalLM.from_pretrained(
    "/home/bob/llama/",
    device_map="auto",  # Automatically map layers to available devices
)

# Resize token embeddings after setting pad_token
model.resize_token_embeddings(len(tokenizer))

# 5. Configure LoRA Adapters
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # For causal language modeling tasks
    inference_mode=False,          # Enable training mode
    r=16,                           # Low-rank adaptation matrix size
    lora_alpha=32,                 # Scaling factor for LoRA
    lora_dropout=0.1,              # Dropout to regularize training
)

# Add LoRA adapters to the base model
model = get_peft_model(model, peft_config)

# 6. Set up training arguments
training_args = TrainingArguments(
    output_dir='./fine-tuned-llama-noquant',
    num_train_epochs=10, #Testing to determine if acceptable loss
    per_device_train_batch_size=1,  # Adjust based on your GPU memory
    max_grad_norm=5.0, #limit gradient magnitude
    gradient_accumulation_steps=4,  # Simulate larger batch sizes
    save_steps=500,
    save_total_limit=2,
    warmup_steps=100,
    logging_steps=50,
    learning_rate=5e-5,
    fp16=False,  # Mixed precision for faster training and lower memory usage, disabled due to instability
    report_to='none',
)

# 7. Initialize Trainer
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
)

# 8. Train the model
trainer.train()

# 9. Save the fine-tuned adapters and tokenizer
model.save_pretrained('./fine-tuned-llama-noquant')
tokenizer.save_pretrained('./fine-tuned-llama-noquant')