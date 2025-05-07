import json, os

def chunk(input_path="data/banking_qa.json", out_dir="rag/chunks/"):
    os.makedirs(out_dir, exist_ok=True)
    with open(input_path) as f:
        qa = json.load(f)
    for i, pair in enumerate(qa):
        # pull out the prompt and response keys for each item in the dictionary.
        doc = f"Q: {pair['prompt']}\nA: {pair['response']}"
        with open(os.path.join(out_dir, f"{i:04d}.txt"), "w") as out:
            out.write(doc)

if __name__ == "__main__":
    chunk()