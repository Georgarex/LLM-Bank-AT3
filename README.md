# 🏦 Banking LLM: Fine-tuned + RAG Assistant

A production-ready framework that combines:

- 🔧 **Fine-tuned LLMs** (GPT-2 & Flan-T5-Small) on bank-specific Q&A  
- 🔎 **Retrieval-Augmented Generation** from real bank documents  
- 🚀 **Gradio** chat UI & **FastAPI** REST endpoint  
- 📊 **Automated evaluation** with ROUGE-L

---

## 📦 Installation

```bash
# create & activate your venv
pip install -r requirements.txt
```

**requirements.txt** should include at least:

```
fastapi
uvicorn[standard]
torch
transformers
datasets
sentence-transformers
faiss-cpu
accelerate>=0.26.0
evaluate
gradio
nltk
```

---

## 🔍 Preprocess / RAG

Chunk, embed, and index your PDF or text files so the retriever can fetch context:

```bash
python rag/chunker.py
```

---

## 🚂 Fine-tuning

### 1. GPT-2 (Causal LM)

```bash
# trains on data/banking_qa.json with batch size 8, 50 epochs
python finetune/train.py
```

### 2. Flan-T5-Small (Seq2Seq LM)

```bash
# trains on data/banking_qa.json with batch size 8, 50 epochs
python finetune/train_flanmini.py
```

Both scripts save checkpoints under `./model/gpt_model` or `./model/flan_model-bs8`.

---

## 📊 Evaluation

Compare your models via ROUGE-L on a held-out split:

```bash
python evaluate_models.py
```

This will print each model’s latest checkpoint and its ROUGE-L F1 score.

---

## 🤖 Run the Chatbot

### Gradio UI

```bash
python gradio_chatbot.py
```

Browse the link it prints (or add `share=True` to expose a public URL).

### FastAPI Endpoint

```bash
uvicorn app:app --reload
```

Then:

```http
POST http://127.0.0.1:8000/query
Content-Type: application/json

{
  "query": "What's the early repayment penalty on a home loan?",
  "use_rag": true,
  "top_k": 3
}
```

---

## 📚 How It Works

1. **RAG** (`rag/chunker.py` + `rag/retriever.py`) splits & indexes your bank docs.  
2. **Fine-tuning** adapts GPT-2 or Flan-T5 to Q&A pairs via `finetune/*.py`.  
3. **Inference** (`prompt_model.py` & `gradio_chatbot.py`) loads the latest checkpoint, optionally fetches context, and generates answers.  
4. **Evaluation** uses ROUGE-L to measure overlap between generated answers and ground-truth responses.

---

Feel free to tweak batch sizes, learning rates, and freezing strategies in `finetune/*.py`—all the training parameters are now aligned so you can compare GPT-2 vs. Flan-T5 head-to-head.
