import gradio as gr
from prompt_model import load_model, generate_response,find_latest_checkpoint
from rag.retriever import get_context

# Mapping between user-friendly names and checkpoint folders
MODEL_OPTIONS = {
    "GPT-2": "./model/gpt_model",
    "DistilGPT-2": "./model/distilgpt2_model"
}

# Cache models
loaded_models = {}
def load_selected_model(model_choice):
    if model_choice not in loaded_models:
        base_dir = MODEL_OPTIONS[model_choice]
        checkpoint_dir = find_latest_checkpoint(base_dir)  # 🔁 automatically find correct subfolder
        tokenizer, model, device = load_model(checkpoint_dir)
        loaded_models[model_choice] = (tokenizer, model, device)
    return loaded_models[model_choice]


# Define chatbot logic
def chat_with_bot(user_input, model_choice):
    tokenizer, model, device = load_selected_model(model_choice)
    context = get_context(user_input)
    return generate_response(tokenizer, model, device, user_input, context)

# Gradio UI
demo = gr.Interface(
    fn=chat_with_bot,
    inputs=[
        gr.Textbox(label="User Input", lines=2, placeholder="Type your banking query..."),
        gr.Dropdown(label="Select Model", choices=["GPT-2", "DistilGPT-2"], value="GPT-2")
    ],
    outputs="text",
    title="Bank Chatbot (GPT-2 vs DistilGPT-2)",
    description="Ask a banking question and compare how each model responds."
)

if __name__ == "__main__":
    demo.launch(share=True)
