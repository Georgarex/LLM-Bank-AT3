import os
import sys
import gc
import gradio as gr
import torch
import threading
import traceback
from prompt_model import load_model, generate_response, find_latest_checkpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Add these imports

# Control memory usage
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

# Memory debugging
DEBUG = True  # Set to False in production

def log_memory():
    """Log memory usage for debugging"""
    if not DEBUG:
        return
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    except ImportError:
        print("psutil not installed - install with 'pip install psutil' for memory debugging")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory: {torch.cuda.memory_allocated(i) / 1024 / 1024:.2f} MB allocated")

# Mapping between user-friendly names and checkpoint folders or model IDs
MODEL_OPTIONS = {
    "GPT-2": "./model/gpt_model",
    "DistilGPT-2 LoRA": "./model/distilgpt2_model",
    "FLAN-T5 Small": "google/flan-t5-small"  # Added FLAN-T5 option
}

# Thread-safe loading
model_lock = threading.Lock()
loaded_models = {}

def load_selected_model(model_choice):
    """Thread-safe model loading with error handling"""
    global loaded_models
    
    with model_lock:
        # If another model is loaded, unload it to save memory
        current_keys = list(loaded_models.keys())
        for key in current_keys:
            if key != model_choice:
                print(f"Unloading model: {key} to save memory")
                del loaded_models[key]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Check if model is already loaded
        if model_choice in loaded_models:
            return loaded_models[model_choice], None
            
        try:
            print(f"Loading model: {model_choice}...")
            log_memory()
            
            # Check if this is FLAN-T5
            if model_choice == "FLAN-T5 Small":
                print("Loading FLAN-T5 directly from Hugging Face")
                model_id = MODEL_OPTIONS[model_choice]
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
                
                # Store with a flag to indicate this is a seq2seq model
                loaded_models[model_choice] = (tokenizer, model, device, True)
                
                print(f"FLAN-T5 loaded successfully on {device}")
                log_memory()
                
                return loaded_models[model_choice], None
            
            # Original code path for GPT models
            base_dir = MODEL_OPTIONS[model_choice]
            if not os.path.exists(base_dir):
                return None, f"Model directory not found: {base_dir}"
                
            try:
                checkpoint_dir = find_latest_checkpoint(base_dir)
                print(f"Found checkpoint: {checkpoint_dir}")
            except FileNotFoundError as e:
                # Try using base directory
                checkpoint_dir = base_dir
                print(f"No checkpoints found, using base directory: {checkpoint_dir}")
            
            # Force CPU for stability
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            
            # Load the model
            tokenizer, model, device = load_model(checkpoint_dir)
            
            # Store with a flag to indicate this is NOT a seq2seq model
            loaded_models[model_choice] = (tokenizer, model, device, False)
            
            print(f"Model loaded successfully: {model_choice}")
            log_memory()
            
            return loaded_models[model_choice], None
        
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error loading model: {str(e)}\n{error_trace}")
            return None, f"Error loading model: {str(e)}"

# Import retriever only when needed to avoid memory issues at startup
def get_rag_context(query):
    """Lazy import and get context from RAG"""
    try:
        # Only import when needed to avoid loading at startup
        from rag.retriever import get_context
        
        print("Getting context from RAG...")
        log_memory()
        
        context = get_context(query)
        
        print(f"Retrieved context (length: {len(context)})")
        log_memory()
        
        return context
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"RAG error: {str(e)}\n{error_trace}")
        return ""

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

def chat_with_bot(user_input, model_choice, use_rag):
    """Generate response with proper error handling and memory management"""
    if not user_input:
        return "Please enter a question."
    
    print("-" * 40)
    print(f"Request: model={model_choice}, use_rag={use_rag}, query='{user_input[:50]}...'")
    
    try:
        # Step 1: Get context if using RAG
        context = ""
        if use_rag:
            context = get_rag_context(user_input)
        
        # Step 2: Load the selected model
        model_data, error = load_selected_model(model_choice)
        if error:
            return f"Error: {error}"
        
        # Unpack model data - now has an is_seq2seq flag as the 4th element
        tokenizer, model, device, is_seq2seq = model_data
        
        # Step 3: Generate response
        print("Generating response...")
        if is_seq2seq:
            # Use FLAN-T5 generation
            response = generate_flan_response(tokenizer, model, device, user_input, context)
        else:
            # Use original GPT generation
            response = generate_response(tokenizer, model, device, user_input, context)
        
        return response
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error generating response: {str(e)}\n{error_trace}")
        
        # Clean up memory on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return f"Error: {str(e)}"
    finally:
        # Always clean up after request
        print("Request completed")
        log_memory()

# Simple UI that doesn't try to load both models at once
with gr.Blocks(title="Banking Chatbot") as demo:
    gr.Markdown("# Banking Chatbot")
    gr.Markdown("Ask a banking question and see how the model responds.")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=list(MODEL_OPTIONS.keys()),
                value=list(MODEL_OPTIONS.keys())[0],
                interactive=True
            )
            rag_checkbox = gr.Checkbox(
                label="Use Retrieval Augmented Generation",
                value=True,
                interactive=True
            )
            user_input = gr.Textbox(
                label="Your Question",
                placeholder="Type your banking question here...",
                lines=3
            )
            submit_btn = gr.Button("Submit")
            
        with gr.Column():
            output = gr.Textbox(
                label="Response",
                lines=10
            )
            
    # Connect the components
    submit_btn.click(
        fn=chat_with_bot,
        inputs=[user_input, model_dropdown, rag_checkbox],
        outputs=output
    )
    
    # Add usage instructions
    gr.Markdown("""
    ## Tips:
    1. Select your preferred model from the dropdown
    2. Toggle RAG on/off to see the impact of retrieval
    3. Type your banking question and click Submit
    
    ⚠️ Note: Models are loaded only when you click Submit to save memory
    """)

if __name__ == "__main__":
    # Print debugging info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Launch with simplified parameters that work across Gradio versions
    try:
        # Try a simple launch first
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860,
            share=False
        )
    except Exception as e:
        print(f"Error during launch: {e}")
        print("Trying alternative launch method...")
        
        # Fallback to even simpler launch
        demo.launch()