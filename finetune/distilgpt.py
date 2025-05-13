# train_lora_distilgpt2.py
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 1) Load DistilGPT-2 and configure pad_token
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# 2) Add LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                     # Rank dimension
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules to apply LoRA
    bias="none",
)

# 3) Apply LoRA adapters to the model
model = get_peft_model(model, lora_config)
print("Trainable parameters:")
model.print_trainable_parameters()  # Print percentage of trainable params

# 4) Preprocess: build input_ids, attention_mask, and labels
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

# 5) Load & tokenize
ds = load_dataset("json", data_files="data/banking_qa.json")["train"]
tokenized = ds.map(
    preprocess,
    batched=False,
    remove_columns=ds.column_names
)

# 6) Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 7) TrainingArguments: 3 epochs, save per epoch
training_args = TrainingArguments(
    output_dir="./model/distilgpt2_lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,    # Can use larger batch with LoRA
    learning_rate=1e-4,               # Higher learning rate works well with LoRA
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=5,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=4,    # Simulate larger batch
)

# 8) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

# 9) Train!
trainer.train()

# 10) Save the final model and tokenizer
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

print("✅ LoRA Training complete — model saved to:", training_args.output_dir)