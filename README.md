# 🏦 Banking LLM: Fine-tuned + RAG Assistant

A production-ready framework that combines:
- 🔧 Fine-tuned LLM on bank-specific FAQs
- 🔎 Retrieval-Augmented Generation from real bank docs
- 🚀 FastAPI serving endpoint

## 🔍 Example Prompt
```json
POST /ask { "query": "What’s the early repayment penalty on a home loan?" }
```

## 📦 Setup
```bash
pip install -r requirements.txt
python prepare_data.py # To the data
python finetune/train.py  # Fine-tune GPT-2
python finetune/distilgpt2_model #Fine-tune Distill GPT model
```

## Run the ui
python gradio_ui.py

## Run benchmarks and evaluations
python evalutate_model.py