from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# 1) Load GPT-2 and configure pad_token
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2) Preprocess: build input_ids, attention_mask, and labels
def preprocess(example):
    text = f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # for causal LM, labels == input_ids
    enc["labels"] = enc["input_ids"].copy()
    return enc

# Load & tokenize
ds = load_dataset("json", data_files="data/banking_qa.json")["train"]
tokenized = ds.map(
    preprocess,
    batched=False,
    remove_columns=ds.column_names
)

# 3) Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 4) TrainingArguments: 3 epochs, save per epoch
training_args = TrainingArguments(
    output_dir="./model/gpt_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=5,
)

# 5) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

# 6) Train!
trainer.train()
print("Training complete — final model in", training_args.output_dir)