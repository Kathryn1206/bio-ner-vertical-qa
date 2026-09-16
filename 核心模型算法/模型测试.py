from transformers import BertTokenizer, BertForTokenClassification
import torch

# ====================== 1. 配置参数（和训练时一致） ======================
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

# ====================== 2. 加载训练好的模型和分词器 ======================
model_path = "C:\\Users\\31755\\exam_bio_final_model"  # 训练好的模型路径
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)
model = model.to(device)
model.eval()  # 预测模式
print(f"✅ 模型加载完成，使用设备：{device}")

# ====================== 3. 复用长文本处理和预测函数 ======================
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

    with torch.no_grad():  # 关闭梯度，节省资源
        for encoded in encoded_chunks:
            outputs = model(
                input_ids=encoded["input_ids"].unsqueeze(0).to(device),
                attention_mask=encoded["attention_mask"].unsqueeze(0).to(device)
            )
            pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
            pred_labels = [id_to_label[id] for id in pred_ids][1:-1]  # 去掉CLS/SEP，已修复.item()问题
            all_pred_labels.extend(pred_labels)

    all_pred_labels = all_pred_labels[:len(list(long_text))]
    return list(long_text), all_pred_labels

# ====================== 4. 测试预测（直接运行） ======================
if __name__ == "__main__":
    # 你想测试的句子
    test_sentence = "今年英语四级报名截止时间是什么"
    # 也可以换其他句子测试
    # test_sentence = "明年教师资格证考试缴费流程是什么"
    
    chars, pred_labels = predict_long_text(test_sentence)
    print(f"\n测试句子：{test_sentence}")
    print("BIO标注结果：")
    for char, label in zip(chars, pred_labels):
        print(f"{char}\t{label}")