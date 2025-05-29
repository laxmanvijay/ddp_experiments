import torch
import gradio as gr
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

def generate_text(prompt, max_length, temperature, top_k, top_p):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

model, tokenizer, device = load_model('./saved_model')

interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=10, maximum=200, value=100, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=50, label="Top K"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top P"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="GPT-2 Text Generation",
    description="Generate text using your trained GPT-2 model"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)