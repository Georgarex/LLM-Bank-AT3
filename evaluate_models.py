#!/usr/bin/env python
"""
evaluate_models.py

Compute and compare performance metrics for models with and without RAG:
- GPT-2-based model
- DistilGPT-2 LoRA model
- FLAN-T5 Small model

Metrics include ROUGE scores, BLEU, exact match, and latency.
"""
import nltk
import evaluate
import torch
import time
import json
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from prompt_model import find_latest_checkpoint, generate_response

# Download punkt for tokenization
nltk.download("punkt", quiet=True)

# Support Flan-T5 response generation
def generate_flan_response(tokenizer, model, device, user_query, context=""):
    """Generate response with FLAN-T5 models"""
    try:
        # Format prompt for FLAN-T5
        if context:
            prompt = f"Answer this banking question with this context: {context}\nQuestion: {user_query}"
        else:
            prompt = f"Answer this banking question: {user_query}"
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        
        # Generate
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=128,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.2
        )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error generating FLAN response: {e}")
        return f"Error: {str(e)}"

# Import RAG functionality
def get_rag_context(query):
    """Get context from RAG system"""
    try:
        # Import only when needed
        from rag.retriever import get_context
        
        print(f"Getting RAG context for: {query[:50]}...")
        context = get_context(query)
        print(f"Retrieved context (length: {len(context)})")
        
        return context
    except Exception as e:
        print(f"RAG error: {str(e)}")
        return ""

def load_and_generate(model_info, test_data, use_rag=False):
    """
    Load model, run inference on test data, and return predictions with metrics.
    
    Args:
        model_info: Tuple of (name, path, is_seq2seq)
        test_data: Dataset to evaluate on
        use_rag: Whether to use RAG
    
    Returns:
        Dictionary with predictions, references, and metrics
    """
    name, model_path, is_seq2seq = model_info
    print(f"\nEvaluating {name} ({'with' if use_rag else 'without'} RAG)...")
    
    # Find the checkpoint
    try:
        if is_seq2seq and 'flan-t5' in model_path.lower():
            # Direct HF model 
            ckpt = model_path
        else:
            # Local checkpoint
            ckpt = find_latest_checkpoint(model_path)
    except FileNotFoundError:
        # Fallback to base directory
        ckpt = model_path

    print(f"Using checkpoint: {ckpt}")
    
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds, refs = [], []
    latencies = []
    
    for ex in test_data:
        query = ex["prompt"]
        reference = ex["response"]
        
        # Get RAG context if needed
        context = ""
        if use_rag:
            context = get_rag_context(query)
        
        # Generate response and measure time
        start_time = time.time()
        
        if is_seq2seq:
            generated = generate_flan_response(tokenizer, model, device, query, context)
        else:
            generated = generate_response(tokenizer, model, device, query, context)
            
        end_time = time.time()
        latency = end_time - start_time
        
        preds.append(generated)
        refs.append(reference)
        latencies.append(latency)
    
    # Calculate metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    # ROUGE scores
    rouge_scores = rouge.compute(
        predictions=preds,
        references=refs,
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )
    
    # BLEU score
    bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
    
    # Exact match percentage
    exact_matches = sum(1 for p, r in zip(preds, refs) if p.strip() == r.strip())
    exact_match_pct = exact_matches / len(preds) if preds else 0
    
    # Average latency
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    # Extract ROUGE F1 scores
    rouge1_f1 = rouge_scores.get("rouge1").mid.fmeasure if hasattr(rouge_scores.get("rouge1"), 'mid') else float(rouge_scores.get("rouge1"))
    rouge2_f1 = rouge_scores.get("rouge2").mid.fmeasure if hasattr(rouge_scores.get("rouge2"), 'mid') else float(rouge_scores.get("rouge2"))
    rougeL_f1 = rouge_scores.get("rougeL").mid.fmeasure if hasattr(rouge_scores.get("rougeL"), 'mid') else float(rouge_scores.get("rougeL"))
    
    # Compile metrics
    metrics = {
        "rouge1_f1": rouge1_f1,
        "rouge2_f1": rouge2_f1, 
        "rougeL_f1": rougeL_f1,
        "bleu": bleu_score["bleu"],
        "exact_match": exact_match_pct,
        "avg_latency": avg_latency,
        "num_samples": len(preds)
    }
    
    return {
        "model": name,
        "rag": use_rag,
        "predictions": preds,
        "references": refs,
        "metrics": metrics
    }

def main():
    # Define model configurations
    models = [
        ("GPT-2", "./model/gpt_model", False),
        ("DistilGPT-2 LoRA", "./model/distilgpt2_model", False),
        ("FLAN-T5 Small", "google/flan-t5-small", True),
    ]
    
    # Load dataset
    print("Loading test dataset...")
    raw = load_dataset("json", data_files="data/banking_qa.json")["train"]
    test_size = 0.1  # Use 10% of data for testing
    seed = 42
    test_split = raw.train_test_split(test_size=test_size, seed=seed)["test"]
    print(f"Test set has {len(test_split)} examples")

    # Results will be stored here
    results = []
    
    # Create results directory
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Run evaluation for each model with and without RAG
    for model_info in models:
        # Without RAG
        no_rag_results = load_and_generate(model_info, test_split, use_rag=False)
        results.append(no_rag_results)
        
        # With RAG
        rag_results = load_and_generate(model_info, test_split, use_rag=True)
        results.append(rag_results)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'RAG':<5} {'ROUGE-1':<8} {'ROUGE-2':<8} {'ROUGE-L':<8} {'BLEU':<8} {'EM%':<6} {'Latency':<8}")
    print("-" * 80)
    
    for result in results:
        model = result["model"]
        rag = "Yes" if result["rag"] else "No"
        metrics = result["metrics"]
        
        print(f"{model:<20} {rag:<5} {metrics['rouge1_f1']:.4f}   {metrics['rouge2_f1']:.4f}   {metrics['rougeL_f1']:.4f}   {metrics['bleu']:.4f}   {metrics['exact_match']*100:.1f}%   {metrics['avg_latency']:.3f}s")
    
    # Save detailed results to JSON
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = f"evaluation_results/eval_results_{timestamp}.json"
    
    # Prepare results for JSON (exclude large prediction lists)
    json_results = []
    for result in results:
        clean_result = {
            "model": result["model"],
            "rag": result["rag"],
            "metrics": result["metrics"],
            # Include a few examples
            "examples": [
                {"prompt": test_split[i]["prompt"], 
                 "reference": result["references"][i], 
                 "prediction": result["predictions"][i]}
                for i in range(min(5, len(result["predictions"])))
            ]
        }
        json_results.append(clean_result)
    
    with open(result_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("\nDetailed evaluation results saved to:", result_file)
    
    # Create CSV summary for easy analysis
    csv_file = f"evaluation_results/summary_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        f.write("Model,RAG,ROUGE-1,ROUGE-2,ROUGE-L,BLEU,ExactMatch,Latency\n")
        for result in results:
            m = result["metrics"]
            f.write(f"{result['model']},{result['rag']},{m['rouge1_f1']:.4f},{m['rouge2_f1']:.4f},{m['rougeL_f1']:.4f},{m['bleu']:.4f},{m['exact_match']:.4f},{m['avg_latency']:.4f}\n")
    
    print(f"Summary CSV saved to: {csv_file}")

if __name__ == "__main__":
    main()