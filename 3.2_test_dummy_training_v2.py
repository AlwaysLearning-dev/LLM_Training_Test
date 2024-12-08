from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from peft import get_peft_model, LoraConfig, TaskType
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('/home/bob/llama/', use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 2. Load the dataset
dataset = load_dataset('json', data_files='rules_dataset.jsonl')

# 3. Prepare the data
def combine_fields(example):
    example['text'] = example['instruction'] + example['output']
    return example

dataset = dataset.map(combine_fields)

# 4. Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=1024, padding='longest')

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Load the base model
model = AutoModelForCausalLM.from_pretrained(
    "/home/bob/llama/",
    device_map="auto",
)

# Resize token embeddings after setting pad_token
model.resize_token_embeddings(len(tokenizer))

# 6. Configure LoRA Adapters
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)

# 7. Define custom learning rate scheduler
def custom_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        # Decay phase (capped at 1e-4)
        return min(1.0, (num_training_steps - current_step) / max(1, num_training_steps - num_warmup_steps))

    return LambdaLR(optimizer, lr_lambda)

# 8. Set up training arguments
training_args = TrainingArguments(
    output_dir='./fine-tuned-llama-noquant',
    num_train_epochs=3,
    per_device_train_batch_size=1,    #I used two for this when it succeeded
    gradient_accumulation_steps=16,
    max_grad_norm=1.0,
    save_steps=500,
    save_total_limit=2,
    warmup_steps=10,  # Gradual warmup
    logging_steps=10,
    learning_rate=1e-4,  # Maximum learning rate
    lr_scheduler_type="cosine",
    fp16=False,
    report_to='none',
)

# 9. Initialize Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
num_training_steps = len(tokenized_dataset['train']) * training_args.num_train_epochs
scheduler = custom_lr_scheduler(optimizer, training_args.warmup_steps, num_training_steps)

# 10. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    optimizers=(optimizer, scheduler),  # Pass initialized optimizer and scheduler
)

# 11. Train the model
trainer.train()

# 12. Save the fine-tuned adapters and tokenizer
model.save_pretrained('./fine-tuned-llama-noquant')
tokenizer.save_pretrained('./fine-tuned-llama-noquant')