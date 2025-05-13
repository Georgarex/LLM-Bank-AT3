import re
import gradio as gr
from prompt_gpt2 import find_latest_checkpoint, load_model, generate_response
from rag.retriever import get_context

# ——— Helpers ——————————————————————————————————————————————————————————————

def clean_text(text: str) -> str:
    """Remove ALL <…> or […] tokens (and stray angle brackets) from the output."""
    # 1) strip out any <...> tags
    text = re.sub(r"<[^>]*>", "", text)
    # 2) strip out any [...]-style tags
    text = re.sub(r"\[[^\]]*\]", "", text)
    # 3) just in case, drop stray < or >
    text = text.replace("<", "").replace(">", "")
    return text.strip()

# ——— Load GPT-2 model ——————————————————————————————————————————————————————

checkpoint_dir = find_latest_checkpoint("./model/gpt_model")
tokenizer, model, device = load_model(checkpoint_dir)

# ——— Gradio interface ——————————————————————————————————————————————————————

def chat_with_bot(user_input: str) -> str:
    # 1) fetch any RAG context
    context = get_context(user_input)
    # 2) run the model
    raw = generate_response(tokenizer, model, device, user_input, context)
    # 3) clean out ALL <…> or […] tokens
    return clean_text(raw)

if __name__ == "__main__":
    demo = gr.Interface(
        fn=chat_with_bot,
        inputs=gr.Textbox(
            label="User Input",
            lines=2,
            placeholder="Type your banking query..."
        ),
        outputs="text",
        title="Bank Chatbot",
        description="Ask any banking-related question!"
    )
    demo.launch(share=True)