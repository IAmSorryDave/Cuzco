import os
import subprocess
import time
import requests
import gradio as gr
from huggingface_hub import InferenceClient, HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "Qwen")
MODEL_PARAMS = os.environ.get("MODEL_PARAMS", "0.5B")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", f"{MODEL_PROVIDER.lower()}2.5-coder:{MODEL_PARAMS}-instruct")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", 11434))

def has_hf_credits(token):
    try:
        api = HfApi(token=token)
        api.whoami()
        client = InferenceClient(token=token)
        _ = client.text_generation("test", model=MODEL_PROVIDER + '/' + OLLAMA_MODEL, max_new_tokens=1)
        return True
    except Exception:
        return False

def start_ollama():
    try:
        r = requests.get(f"http://localhost:{OLLAMA_PORT}/api/tags", timeout=2)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    subprocess.Popen(["ollama", "serve"])
    for _ in range(10):
        try:
            r = requests.get(f"http://localhost:{OLLAMA_PORT}/api/tags", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    return False

def chat_interface(message, history):
    prompt = message
    # Try Hugging Face first
    if HF_TOKEN and has_hf_credits(HF_TOKEN):
        try:
            client = InferenceClient(token=HF_TOKEN)
            result = client.text_generation(prompt, model=MODEL_PROVIDER + '/' + OLLAMA_MODEL, max_new_tokens=128)
            return result
        except Exception:
            pass
    # Fallback to Ollama
    if start_ollama():
        try:
            r = requests.post(f"http://localhost:{OLLAMA_PORT}/api/generate", json={"model": OLLAMA_MODEL, "prompt": prompt})
            if r.ok:
                return r.json().get("response", "")
            else:
                return f"Ollama error: {r.text}"
        except Exception as e:
            return f"Ollama not available: {str(e)}"
    return "No inference provider available."

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(chat_interface, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
