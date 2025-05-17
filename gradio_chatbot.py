# gradio_chatbot.py

import gradio as gr
from prompt_model import find_latest_checkpoint, load_model, generate_response
from rag.retriever import get_context

# Load the model once (instead of every request)
checkpoint_dir = find_latest_checkpoint("./model/gpt_model")
tokenizer, model, device = load_model(checkpoint_dir)

def chat_with_bot(user_input):
    context = get_context(user_input)
    return generate_response(tokenizer, model, device, user_input, context)

if __name__ == "__main__":
    demo = gr.Interface(
        fn=chat_with_bot,
        inputs=gr.Textbox(label="User Input", lines=2, placeholder="Type your banking query..."),
        outputs="text",
        title="Bank Chatbot",
        description="Ask any banking-related question!"
    )
    demo.launch(share=True)  # <-- this line is the fix

