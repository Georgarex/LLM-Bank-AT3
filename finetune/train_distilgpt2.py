# finetune/train_distilgpt2.py
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import json
import torch

# Load distilgpt2
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Load dataset
with open("data/banking_qa.json", "r") as f:
    raw_data = json.load(f)
ds = Dataset.from_list(raw_data)

def preprocess(example):
    text = f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"
    enc = tokenizer(
        text, truncation=True, padding="max_length", max_length=512
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized = ds.map(preprocess, batched=False, remove_columns=ds.column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./model/distilgpt2_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=5,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

trainer.train()

print("✅ Training complete — model saved to:", training_args.output_dir)
