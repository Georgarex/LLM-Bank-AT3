import nltk
import numpy as np
import torch
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset

# Download punkt for ROUGE tokenization
nltk.download("punkt")

# 1) Configuration
MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = "./model/flan_model-bs8" # change dir if unfreeze/ not unfreeze
BATCH_SIZE = 8
EPOCHS = 50
MAX_LENGTH = 256

# 2) Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

## COMMENT OUT IF NOT FREEZE LAYERS - PREVENTS OVERFITTING ESPECIALLY WITH SMALLER DATASET...
# 2.1) Freeze all layers except the final lm_head
# for param in model.parameters():
#     param.requires_grad = False
# if hasattr(model, "lm_head"):
#     for param in model.lm_head.parameters():
#         param.requires_grad = True

model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 3) Load and split dataset
raw = load_dataset("json", data_files="data/banking_qa.json")["train"]
split = raw.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]

# 4) Preprocessing for seq2seq

def preprocess(example):
    # Format input and target
    prompt = f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"
    # Split prompt and response
    parts = prompt.split(" [/INST] ")
    input_text = parts[0] + " [/INST]"
    target_text = parts[1]

    inputs = tokenizer(
        input_text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
        )
    inputs["labels"] = labels["input_ids"].copy()
    return inputs

# Tokenize datasets
train_tkn = train_ds.map(
    preprocess,
    batched=False,
    remove_columns=train_ds.column_names,
)
eval_tkn = eval_ds.map(
    preprocess,
    batched=False,
    remove_columns=eval_ds.column_names,
)

# 5) Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
)

# 6) Load ROUGE metric
rouge = evaluate.load("rouge")

# 7) Metrics function
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # Generate predictions if needed
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rougeL"],
    )
    return {"rougeL_f1": result["rougeL"].mid.fmeasure}

# 8) Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

# 9) Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tkn,
    eval_dataset=eval_tkn,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 10) Train
print(f"🔄 Fine-tuning {MODEL_NAME} → saving to {OUTPUT_DIR}")
trainer.train()
print("✅ Done. Checkpoints at", OUTPUT_DIR)
