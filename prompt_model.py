# prompt_model.py

import os, glob, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def find_latest_checkpoint(base_dir="./model/gpt_model"):
    # look for folders named checkpoint-*
    paths = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")
    # sort by the numeric suffix
    paths.sort(key=lambda p: int(p.rsplit("-",1)[-1]))
    return paths[-1]

def load_model(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def generate_response(tokenizer, model, device, user_query, context=""):
    prompt = f"<s>[INST] {context}\n{user_query} [/INST]"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # extract only the answer part
    return decoded.split("[/INST]")[-1].strip()

def main():
    ckpt = find_latest_checkpoint()
    tokenizer, model, device = load_model(ckpt)

    print(">> Loaded checkpoint:", ckpt)
    print(">> Type your query and press Enter. (type 'exit' or Ctrl-C to quit)\n")

    try:
        while True:
            query = input("User: ").strip()
            if not query or query.lower() in ("exit", "quit"):
                break
            # if you have a RAG retriever, call it here to get `context`
            context = ""
            response = generate_response(tokenizer, model, device, query, context)
            print(f"Assistant: {response}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")

if __name__ == "__main__":
    main()