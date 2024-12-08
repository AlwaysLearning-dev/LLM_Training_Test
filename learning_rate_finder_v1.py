import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
import torch

# 1. Load the dataset
dataset = load_dataset('json', data_files='dummy_dataset_randomized.jsonl')

# 2. Prepare the data
def combine_fields(example):
    example['text'] = example['input'] + example['output']
    return example

dataset = dataset.map(combine_fields)

# 3. Tokenize the data
tokenizer = AutoTokenizer.from_pretrained('/home/bob/gemma/', use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512, padding='longest')

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. Load the base model (without quantization)
model = AutoModelForCausalLM.from_pretrained(
    "/home/bob/gemma/",
    device_map="auto",
)
model.resize_token_embeddings(len(tokenizer))

# 5. Configure LoRA Adapters
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)

# 6. Set up the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. Define learning rate finder function
def learning_rate_finder(trainer, lr_start=1e-7, lr_end=1e-3, num_steps=100):
    """
    Gradually increases the learning rate during training to find the optimal range.
    
    Args:
        trainer (Trainer): Hugging Face Trainer instance.
        lr_start (float): Initial learning rate.
        lr_end (float): Maximum learning rate.
        num_steps (int): Number of steps over which to test.
    
    Returns:
        list: Learning rates tested.
        list: Corresponding losses.
    """
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=lr_start)
    lr_values = []
    losses = []

    # Exponential learning rate scheduler
    def update_lr(step):
        return lr_start * (lr_end / lr_start) ** (step / num_steps)

    trainer.model.train()
    for step in range(num_steps):
        batch = next(iter(trainer.get_train_dataloader()))
        outputs = trainer.model(**batch)
        loss = outputs.loss

        # Record learning rate and loss
        lr_values.append(update_lr(step))
        losses.append(loss.item())

        # Update learning rate and optimizer step
        for param_group in optimizer.param_groups:
            param_group['lr'] = update_lr(step)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return lr_values, losses

# 8. Initialize Trainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./fine-tuned-gemma-noquant',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    num_train_epochs=1,  # One epoch for learning rate testing
    logging_steps=50,
    save_steps=500,
    warmup_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
)

# 9. Run learning rate finder
lr_start = 1e-7
lr_end = 1e-3
num_steps = 100
lr_values, losses = learning_rate_finder(trainer, lr_start, lr_end, num_steps)

# 10. Plot the results
plt.plot(lr_values, losses)
plt.xscale('log')
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate Finder")
plt.show()