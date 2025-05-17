# 🏦 Banking LLM: Fine-tuned + RAG Assistant

Models are in model/
Evaluation charts and results are in evaluation_results/

A production-ready framework that combines:

## 📦 Setup

### Prerequisites
- Python 3.8+ 
- pip
- Virtual environment (recommended)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/banking-llm.git
cd banking-llm
```

2. Set up a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
# create & activate your venv
pip install -r requirements.txt
```

### Data Preparation
```bash
# Prepare the data for the RAG system
python prepare_data.py
```

### Model Training
```bash
# Fine-tune GPT-2 model
python finetune/train.py

# Fine-tune DistilGPT-2 model with LoRA
python finetune/distilgpt2_model.py
```

## 🚀 Running the Application

### Run the Web UI
```bash
python gradio_ui.py
```

## 📊 Evaluation and Benchmarking

Run performance evaluation across models:
```bash
python evaluate_models.py
```

Generate visualization of model comparisons only works after running evaluate_models.pys:
```bash
python model_comparison_visualize.py
```
## 📁 Project Structure

- `app.py`: FastAPI server
- `gradio_ui.py`: Gradio-based web interface
- `evaluate_models.py`: Benchmarking script for model performance
- `model_comparison_visualize.py`: Generate performance visualizations
- `prepare_data.py`: Prepare data for the RAG system
- `finetune/`: Model fine-tuning scripts
- `rag/`: Retrieval-Augmented Generation components
  - `chunker.py`: Document chunking
  - `retriever.py`: Vector search and retrieval
