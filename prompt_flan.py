# prompt_model.py

import os
import glob
import sys
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from rag.retriever import get_context


def find_latest_checkpoint(base_dir: str):
    paths = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")
    paths.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
    return paths[-1]


def load_model(checkpoint_dir: str):
    # detect model type from config
    config    = AutoConfig.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(config, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def generate_response(
    tokenizer,
    model,
    device,
    user_query: str,
    use_rag: bool = False,
    top_k: int   = 3,
) -> str:
    context = get_context(user_query, top_k=top_k) if use_rag else ""
    prompt  = f"<s>[INST] {context}\n{user_query} [/INST]"
    inputs  = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full.split("[/INST]")[-1].strip()


def main():
    # allow passing a different base_dir on the cmd-line
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "./model/flan_model" # change this line
    ckpt     = find_latest_checkpoint(base_dir)
    tokenizer, model, device = load_model(ckpt)

    print(f">> Loaded checkpoint: {ckpt}", flush=True)
    print(">> Type your query (or ‘exit’ to quit)\n", flush=True)

    try:
        while True:
            q = input("User: ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break
            ans = generate_response(tokenizer, model, device, q, use_rag=True)
            print("Assistant:", ans, "\n", flush=True)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.", flush=True)


if __name__ == "__main__":
    main()