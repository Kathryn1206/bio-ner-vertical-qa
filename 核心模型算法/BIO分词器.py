from transformers import BertTokenizer,Trainer, TrainingArguments, BertForTokenClassification
import torch
import numpy as np
from transformers import BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
import os
labels = [
    'O',
    'B-EXAM', 'I-EXAM',
    'B-SUBJECT', 'I-SUBJECT',
    'B-TIME', 'I-TIME',
    'B-ACTION', 'I-ACTION',
    'B-CONSTRAINT', 'I-CONSTRAINT'
]

EXAM = [
    "计算机考试",
    "研究生考试",
    "英语四级",
    "英语六级",
    "教师资格证"
]

ACTION = [
    "报名",
    "缴费",
    "考试",
    "查询成绩",
    "打印准考证"
]

TIME = [
    "今年",
    "明年",
    "去年",
    "本月",
    "上半年",
    "下半年"
]

CONSTRAINT = [
    "时间",
    "截止时间",
    "条件",
    "流程",
    "费用"
]

label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {i: label for i, label in enumerate(labels)}  
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f'模型已加载至{device}')
max_length = 510  # BERT最大输入长度512，留出[CLS]和[SEP]位置

#长文本处理函数
def process_long_text(long_text, long_labels=None, max_length=510):
    text_chars = list(long_text)
    total_chars = len(text_chars) # 长文本拆分成单字
    chunks = [] # 分段后文字列表
    label_chunks = [] #分段后标签列表
    encoded_chunks = []
    processed_labels = []
    raw_chunks = []


    for i in range(0, total_chars, max_length):
        chunk = text_chars[i:i+max_length]
        chunks.append(chunk) #切文本
        if long_labels is not None:  # 切标签
            label_chunk = long_labels[i:i+max_length]
            if len(label_chunk) < max_length: #补齐标签
                label_chunk += ['O'] * (max_length - len(label_chunk))
            label_chunks.append(label_chunk)

    # 对每个文本段进行BERT编码
    for chunk in chunks:
        encoding =tokenizer(
            chunk,
            is_split_into_words = True,
            return_tensors='pt',
            padding = 'max_length',
            truncation = True,
            max_length=min(max_length + 2, 512)    #考虑[CLS]和[SEP],强制限制512以内
        )
        encoded_chunks.append({
                'input_ids': encoding['input_ids'].squeeze(0), 
                'attention_mask': encoding['attention_mask'].squeeze(0) 
            })

    
    # 处理标签段
    if long_labels is not None:
         for idx, label_chunk in enumerate(label_chunks):  # 枚举遍历
            label_ids = [label_to_id[l] for l in label_chunk]  # 转数字
            label_ids = [-100] + label_ids + [-100]  # 给 [CLS] 和 [SEP] 添加 -100
            processed_labels.append(torch.tensor(label_ids))  # 转张量

    return encoded_chunks, processed_labels, chunks

#长文本预测函数
def predict_long_text(long_text, max_length = 510):
    encoded_chunks, _, text_chunks = process_long_text(long_text, None, max_length=max_length)
    all_pred_labels = [] # 存储所有预测标签
    id_to_label = {i: label for label, i in label_to_id.items()} # 数字标签转回文本标签
    model.eval()
    with torch.no_grad():  # 关闭梯度，节省资源
        for encoded in encoded_chunks: # 遍历每个文本段
            outputs = model( 
                input_ids=encoded["input_ids"].unsqueeze(0).to(device),  # 加batch维度
                attention_mask=encoded["attention_mask"].unsqueeze(0).to(device)
            )
            pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()# 取预测结果（去掉[CLS]和[SEP]）
            pred_labels = [id_to_label[id] for id in pred_ids][1:-1]  # 去掉首尾的[CLS]/[SEP]
            all_pred_labels.extend(pred_labels)
    all_pred_labels = all_pred_labels[:len(list(long_text))]  # 截断多余部分
    return list(long_text), all_pred_labels



#考试提问bio编码自动生成
def bio_tag_sentence(sentence, entities):
    labels = []
    i = 0
    while i < len(sentence):
        matched = False
        for ent_text, ent_type in entities:
            if sentence.startswith(ent_text, i):
                labels.append(f"B-{ent_type}")
                for _ in range(len(ent_text) - 1):
                    labels.append(f"I-{ent_type}")
                i += len(ent_text)
                matched = True
                break
        if not matched:
            labels.append("O")
            i += 1
    return labels


def generate_dataset():
    dataset = []

    templates = [
        "{TIME}{EXAM}{ACTION}{CONSTRAINT}",
        "{EXAM}{ACTION}{CONSTRAINT}",
        "{TIME}{EXAM}{ACTION}是什么",
        "{EXAM}{ACTION}什么时候",
        "{EXAM}{ACTION}截止时间"
    ]

    for exam in EXAM:
        for action in ACTION:
            for time in TIME:
                for constraint in CONSTRAINT:
                    for template in templates:
                        sentence = template.format(
                            TIME=time,
                            EXAM=exam,
                            ACTION=action,
                            CONSTRAINT=constraint
                        )

                        entities = [
                            (time, "TIME"),
                            (exam, "EXAM"),
                            (action, "ACTION"),
                            (constraint, "CONSTRAINT")
                        ]

                        labels = bio_tag_sentence(sentence, entities)
                        dataset.append((sentence, labels))

    return dataset

# 自定义数据集类（适配你的长文本处理+标签格式）
class ExamBIODataset(Dataset):
    def __init__(self, data, tokenizer, label_to_id, max_length=510):
        self.data = data  # 格式：[(sentence, labels), ...]
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, labels = self.data[idx]
        
        # 调用你的长文本处理函数（兼容长短文本）
        encoded_chunks, processed_labels, _ = process_long_text(
            long_text=sentence,
            long_labels=labels,
            max_length=self.max_length
        )
        
        # 取第一个chunk（考试提问都是短文本，不会超过510）
        encoded = encoded_chunks[0]
        label_ids = processed_labels[0]
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": label_ids
        }

# 评估指标函数（训练中监控F1/准确率）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # 过滤-100（[CLS]/[SEP]的标签）
    true_labels = []
    true_predictions = []
    for pred_row, label_row in zip(predictions, labels):
        for p, l in zip(pred_row, label_row):
            if l != -100:
                true_labels.append(id_to_label[l])
                true_predictions.append(id_to_label[p])
    
    # 计算核心指标
    from sklearn.metrics import classification_report
    report = classification_report(
    true_labels,
    true_predictions,
    output_dict=True,
    zero_division=0
    )
    return {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"]
    }

# 训练主函数
def train_model():
    # 1. 生成BIO数据集
    print("===== 生成考试提问BIO数据集 =====")
    raw_dataset = generate_dataset()
    print(f"数据集总数：{len(raw_dataset)} 条")
    print("示例数据：", raw_dataset[0])  # 验证生成结果
    
    # 2. 拆分训练集/验证集（8:2）
    train_size = int(0.8 * len(raw_dataset))
    train_data = raw_dataset[:train_size]
    val_data = raw_dataset[train_size:]
    
    # 3. 构建数据集
    train_dataset = ExamBIODataset(train_data, tokenizer, label_to_id, max_length)
    val_dataset = ExamBIODataset(val_data, tokenizer, label_to_id, max_length)
    
    # 4. 配置训练参数
    training_args = TrainingArguments(
        output_dir="./exam_bio_model",  # 模型保存路径
        num_train_epochs=10,            # 训练轮数（短文本数据集10轮足够）
        per_device_train_batch_size=8,  # 批次大小
        per_device_eval_batch_size=8,   # 验证批次大小
        learning_rate=2e-5,             # 学习率（BERT最优区间）
        logging_dir="./exam_bio_logs",  # 日志路径
        logging_steps=50,               # 每50步打印日志
        evaluation_strategy="epoch",    # 每个epoch评估一次
        save_strategy="epoch",          # 每个epoch保存模型
        load_best_model_at_end=True,    # 训练结束加载最优模型
        metric_for_best_model="f1_macro",  # 按F1选最优模型
        fp16=True,                      # 混合精度训练
        report_to="none",               # 不使用wandb
        remove_unused_columns=False,    # 保留自定义字段
        no_cuda=False,                  # 强制使用GPU
        dataloader_pin_memory=False,    # 关闭pin_memory避免部分环境报错
        dataloader_num_workers=0,     # 关闭多线程避免部分环境报错 
        gradient_accumulation_steps=4   # 累积梯度
)
    
    # 5. 构建Trainer并训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # 执行训练
    print("===== 开始训练 =====")
    trainer.train()
    
    # 6. 保存最终模型
    print("===== 训练完成，保存模型 =====")
    model.save_pretrained("./exam_bio_final_model")
    tokenizer.save_pretrained("./exam_bio_final_model")
    
    # 7. 测试预测（验证效果）
    test_sentence = "今年英语四级报名截止时间是什么"
    chars, pred_labels = predict_long_text(test_sentence)
    print(f"\n测试句子：{test_sentence}")
    print("BIO标注结果：")
    for char, label in zip(chars, pred_labels):
        print(f"{char}\t{label}")

# 启动训练
if __name__ == "__main__":
    train_model()

