import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag.retriever import get_context


def find_latest_checkpoint(base_dir: str = "./model/gpt_model") -> str:
    """
    Scan `base_dir` for subfolders named `checkpoint-*` and return the latest one.
    """
    paths = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")
    # Sort by numeric suffix after the last '-'
    paths.sort(key=lambda p: int(p.rsplit('-', 1)[-1]))
    return paths[-1]


def load_model(checkpoint_dir: str):
    """
    Load tokenizer and model from a local checkpoint directory.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    top_k: int = 3
) -> str:
    """
    Build a prompt, optionally retrieve context via RAG, and generate a response.

    Args:
      - tokenizer, model, device: loaded objects
      - user_query: the user's question string
      - use_rag: whether to fetch context from RAG
      - top_k: number of RAG docs to retrieve
    Returns:
      - Generated answer text
    """
    # 1) Optionally fetch retrieval context
    context = ""
    if use_rag:
        context = get_context(user_query, top_k=top_k)

    # 2) Format prompt for our instruction-style model
    prompt = f"<s>[INST] {context}\n{user_query} [/INST]"

    # 3) Tokenize and move to device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    # 4) Generate
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    # 5) Decode and strip off prompt
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Return only the part after the [/INST] tag
    return full_text.split("[/INST]")[-1].strip()


def main():
    """
    Simple REPL to test generate_response interactively.
    """
    ckpt = find_latest_checkpoint()
    tokenizer, model, device = load_model(ckpt)

    print(f">> Loaded checkpoint: {ckpt}")
    print(">> Type your query and press Enter. (type 'exit' or Ctrl-C to quit)\n")

    try:
        while True:
            query = input("User: ").strip()
            if not query or query.lower() in ("exit", "quit"):
                break
            answer = generate_response(
                tokenizer,
                model,
                device,
                query,
                use_rag=True,
                top_k=3
            )
            print(f"Assistant: {answer}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


if __name__ == "__main__":
    main()
