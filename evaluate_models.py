#!/usr/bin/env python
"""
evaluate_models.py

Compute and compare ROUGE-L F1 scores for two fine-tuned models:
- GPT-2-based model (causal LM)
- Flan-T5 small model (seq2seq LM)
"""
import nltk
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from prompt_gpt2 import find_latest_checkpoint, generate_response

# Download punkt for ROUGE tokenization
nltk.download("punkt", quiet=True)


def load_and_generate(parent_dir, is_seq2seq=False, test_size=0.1, seed=42):
    """
    Load the latest checkpoint under parent_dir, run inference on test split,
    and return lists of (prediction, reference).
    """
    # find the newest subfolder checkpoint-*
    ckpt = find_latest_checkpoint(parent_dir)

    # load dataset and split
    raw = load_dataset("json", data_files="data/banking_qa.json")["train"]
    test_split = raw.train_test_split(test_size=test_size, seed=seed)["test"]

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds, refs = [], []
    for ex in test_split:
        prompt = ex["prompt"]
        reference = ex["response"]
        # generate (empty RAG context)
        generated = generate_response(tokenizer, model, device, prompt, "")
        preds.append(generated)
        refs.append(reference)

    return preds, refs


def main():
    rouge = evaluate.load("rouge")

    # parent directories and flags
    models = [
    ("GPT2-finetuned", "./model/gpt_model", False),  # causal LM directory
    ("Flan-T5-small", "./model/flan_model-bs8", True),  # seq2seq LM directory
]

    for name, parent_dir, is_seq2seq in models:
        print(f"Evaluating {name} ({parent_dir}) ...")
        # run generate
        preds, refs = load_and_generate(parent_dir, is_seq2seq=is_seq2seq)
        # compute ROUGE-L
        scores = rouge.compute(
            predictions=preds,
            references=refs,
            rouge_types=["rougeL"],
        )
        # extract F1 (lib may return float or Score)
        rougeL = scores.get("rougeL")
        if hasattr(rougeL, 'mid'):
            f1 = rougeL.mid.fmeasure
        else:
            f1 = float(rougeL)

        # print results
        ckpt = find_latest_checkpoint(parent_dir)
        print(f"{name} latest checkpoint: {ckpt}")
        print(f"{name} ROUGE-L F1: {f1:.4f}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
