from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

def preprocess(example):
    prompt = f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"
    return tokenizer(prompt, truncation=True)

data = load_dataset("json", data_files="data/bank_faq.json")
tokenized = data.map(preprocess)

args = TrainingArguments(
    output_dir="./model/mistral_model",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"]
)

trainer.train()
