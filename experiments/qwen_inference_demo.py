import torch
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, AutoModelForCausalLM, AutoTokenizer
from fuzzywuzzy import process
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
model_name = os.getenv("QWEN_MODEL_PATH", str(PROJECT_ROOT / "Qwen"))
device = "cuda" if torch.cuda.is_available() else "cpu"  # 检测GPU
print(f"已加载至: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # 自动分配到 GPU/CPU
    trust_remote_code=True,     # Qwen 需要此参数
    torch_dtype=torch.float16 if device == "cuda" else torch.float32  # GPU 用半精度提速
).to(device)

model.eval()  # 切换评估模式
def chat_with_qwen(prompt: str, history=None):
    """与 Qwen-1.8B-Chat 对话"""
    if history is None:
        history = []
    
    # Qwen-1.8B-Chat 要求使用 `chat` 方法（由官方 custom code 实现）
    response, history = model.chat(tokenizer, prompt, history=history)
    return response, history

# 示例：单轮对话
if __name__ == "__main__":
    user_input = "你好！你是谁？"
    response, _ = chat_with_qwen(user_input)
    print("Qwen:", response)

    # 多轮示例
    history = []
    while True:
        user_input = input("你: ")
        if user_input.lower() in ["退出", "quit", "exit"]:
            break
        response, history = chat_with_qwen(user_input, history)
        print("Qwen:", response)

