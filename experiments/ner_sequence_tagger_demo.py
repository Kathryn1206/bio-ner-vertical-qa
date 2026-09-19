from transformers import BertTokenizer, BertForTokenClassification
import torch
import os
from pathlib import Path

# 1. Configuration shared with training
labels = [
    'O',
    'B-EXAM', 'I-EXAM',
    'B-SUBJECT', 'I-SUBJECT',
    'B-TIME', 'I-TIME',
    'B-ACTION', 'I-ACTION',
    'B-CONSTRAINT', 'I-CONSTRAINT'
]
label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {i: label for i, label in enumerate(labels)}
max_length = 510   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load the trained model and tokenizer
PROJECT_ROOT = Path(__file__).resolve().parents[1]
model_path = os.getenv(
    "NER_MODEL_PATH", str(PROJECT_ROOT / "exam_ner_model")
)  # Trained BERT NER model directory
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)
model = model.to(device)
model.eval()  # Inference mode
print(f"Model loaded on: {device}")

# 3. Long-text preprocessing and inference
def process_long_text(long_text, long_labels=None, max_length=128):
    text_chars = list(long_text)
    total_chars = len(text_chars)
    chunks = []
    encoded_chunks = []

    for i in range(0, total_chars, max_length):
        chunk = text_chars[i:i+max_length]
        chunks.append(chunk)

    for chunk in chunks:
        encoding = tokenizer(
            chunk,
            is_split_into_words=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=min(max_length + 2, 512)
        )
        encoded_chunks.append({
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        })

    return encoded_chunks, None, chunks

def predict_long_text(long_text, max_length=128):
    encoded_chunks, _, text_chunks = process_long_text(long_text, None, max_length=max_length)
    all_pred_labels = []

    with torch.no_grad():  # Disable gradients during inference.
        for encoded in encoded_chunks:
            outputs = model(
                input_ids=encoded["input_ids"].unsqueeze(0).to(device),
                attention_mask=encoded["attention_mask"].unsqueeze(0).to(device)
            )
            pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
            pred_labels = [id_to_label[id] for id in pred_ids][1:-1]  # Remove CLS and SEP labels.
            all_pred_labels.extend(pred_labels)

    all_pred_labels = all_pred_labels[:len(list(long_text))]
    return list(long_text), all_pred_labels

# 4. Standalone inference example
if __name__ == "__main__":
    # Replace this query with another Chinese exam question as needed.
    test_sentence = "今年英语四级报名截止时间是什么"
    # Alternative example:
    # test_sentence = "明年教师资格证考试缴费流程是什么"
    
    chars, pred_labels = predict_long_text(test_sentence)
    print(f"\nTest sentence: {test_sentence}")
    print("B–I–O labels:")
    for char, label in zip(chars, pred_labels):
        print(f"{char}\t{label}")
