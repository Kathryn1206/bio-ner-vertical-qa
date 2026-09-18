import torch
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, AutoModelForCausalLM, AutoTokenizer
from fuzzywuzzy import process
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
model_name = os.getenv("QWEN_MODEL_PATH", str(PROJECT_ROOT / "Qwen"))
device = "cuda" if torch.cuda.is_available() else "cpu"  # Prefer CUDA when available.
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # Distribute weights across available devices.
    trust_remote_code=True,     # Required by this Qwen implementation.
    torch_dtype=torch.float16 if device == "cuda" else torch.float32  # Use FP16 on CUDA.
).to(device)

model.eval()  # Inference mode
def chat_with_qwen(prompt: str, history=None):
    """Generate a Qwen-1.8B-Chat response while preserving conversation history."""
    if history is None:
        history = []
    
    # Qwen-1.8B-Chat exposes its conversation API through trusted custom code.
    response, history = model.chat(tokenizer, prompt, history=history)
    return response, history

# Single-turn example
if __name__ == "__main__":
    user_input = "你好！你是谁？"
    response, _ = chat_with_qwen(user_input)
    print("Qwen:", response)

    # Multi-turn example
    history = []
    while True:
        user_input = input("你: ")
        if user_input.lower() in ["退出", "quit", "exit"]:
            break
        response, history = chat_with_qwen(user_input, history)
        print("Qwen:", response)

