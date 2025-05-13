import numpy as np
import torch
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# 1) Configuration
BATCH_SIZE = 8
EPOCHS = 50
MAX_LENGTH = 256

# 2) Load GPT-2 and set pad token
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# ─── FREEZE ALL LAYERS EXCEPT THE FINAL LM HEAD ─────────────────────────────────

# for param in model.parameters():
#     param.requires_grad = False
#
# # unfreeze just the lm_head
# for param in model.lm_head.parameters():
#     param.requires_grad = True

# ────────────────────────────────────────────────────────────────────────────────

# 3) Dataset split

# 3) Dataset split
raw = load_dataset("json", data_files="data/banking_qa.json")["train"]
split = raw.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]

# 4) Preprocessing function
def preprocess(ex):
    text = f"<s>[INST] {ex['prompt']} [/INST] {ex['response']}</s>"
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized_train = train_ds.map(preprocess, batched=False, remove_columns=train_ds.column_names)
tokenized_eval  = eval_ds.map(preprocess,  batched=False, remove_columns=eval_ds.column_names)

# 5) Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 6) ROUGE-L metric
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    res = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rougeL"],
    )
    val = res["rougeL"]
    try:
        f1 = val.mid.fmeasure
    except:
        f1 = float(val)
    return {"rougeL_f1": f1}

# 7) TrainingArguments (match flan-t5 settings)
training_args = TrainingArguments(
    output_dir             = "./model/gpt_model",
    num_train_epochs       = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    # evaluation_strategy    = "epoch", ## not a parameter
    save_strategy          = "epoch",
    logging_steps          = 10,
    save_total_limit       = 3,
    fp16                   = torch.cuda.is_available(),
)

# 8) Trainer
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = tokenized_train,
    eval_dataset    = tokenized_eval,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

# 9) Train
print(f"🔄 Fine-tuning GPT-2 → saving to {training_args.output_dir}")
trainer.train()
print(f"✅ GPT-2 fine-tuning complete — checkpoints in {training_args.output_dir}")
