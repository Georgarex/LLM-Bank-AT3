from fastapi import FastAPI, Request
from rag.retriever import get_context
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

model = AutoModelForCausalLM.from_pretrained("./model/mistral_model")
tokenizer = AutoTokenizer.from_pretrained("./model/mistral_model")

def generate(prompt, context):
    full_prompt = f"<s>[INST] {context}\n{prompt} [/INST]"
    tokens = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    output = model.generate(**tokens, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.post("/ask")
async def ask(req: Request):
    body = await req.json()
    query = body.get("query")
    context = get_context(query)
    return {"response": generate(query, context)}