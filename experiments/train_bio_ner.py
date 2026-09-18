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
print(f'Model loaded on: {device}')
max_length = 510  # Reserve two positions in BERT's 512-token limit for CLS and SEP.

# Split long text into BERT-compatible chunks.
def process_long_text(long_text, long_labels=None, max_length=510):
    text_chars = list(long_text)
    total_chars = len(text_chars)
    chunks = []
    label_chunks = []
    encoded_chunks = []
    processed_labels = []
    raw_chunks = []


    for i in range(0, total_chars, max_length):
        chunk = text_chars[i:i+max_length]
        chunks.append(chunk)
        if long_labels is not None:
            label_chunk = long_labels[i:i+max_length]
            if len(label_chunk) < max_length:  # Pad the final label chunk.
                label_chunk += ['O'] * (max_length - len(label_chunk))
            label_chunks.append(label_chunk)

    # Encode each chunk independently.
    for chunk in chunks:
        encoding =tokenizer(
            chunk,
            is_split_into_words = True,
            return_tensors='pt',
            padding = 'max_length',
            truncation = True,
            max_length=min(max_length + 2, 512)  # Include CLS and SEP without exceeding 512.
        )
        encoded_chunks.append({
                'input_ids': encoding['input_ids'].squeeze(0), 
                'attention_mask': encoding['attention_mask'].squeeze(0) 
            })

    
    # Convert BIO labels to training IDs.
    if long_labels is not None:
         for idx, label_chunk in enumerate(label_chunks):
            label_ids = [label_to_id[l] for l in label_chunk]
            label_ids = [-100] + label_ids + [-100]  # Ignore CLS and SEP in the loss.
            processed_labels.append(torch.tensor(label_ids))

    return encoded_chunks, processed_labels, chunks

# Run token classification over one or more chunks.
def predict_long_text(long_text, max_length = 510):
    encoded_chunks, _, text_chunks = process_long_text(long_text, None, max_length=max_length)
    all_pred_labels = []
    id_to_label = {i: label for label, i in label_to_id.items()}
    model.eval()
    with torch.no_grad():
        for encoded in encoded_chunks:
            outputs = model(
                input_ids=encoded["input_ids"].unsqueeze(0).to(device),  # Add the batch dimension.
                attention_mask=encoded["attention_mask"].unsqueeze(0).to(device)
            )
            pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
            pred_labels = [id_to_label[id] for id in pred_ids][1:-1]  # Remove CLS and SEP labels.
            all_pred_labels.extend(pred_labels)
    all_pred_labels = all_pred_labels[:len(list(long_text))]
    return list(long_text), all_pred_labels



# Generate character-level BIO labels for synthetic exam questions.
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

# Dataset wrapper for the chunked text and BIO-label format.
class ExamBIODataset(Dataset):
    def __init__(self, data, tokenizer, label_to_id, max_length=510):
        self.data = data  # Expected format: [(sentence, labels), ...]
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, labels = self.data[idx]
        
        # Use the same preprocessing path for short and long inputs.
        encoded_chunks, processed_labels, _ = process_long_text(
            long_text=sentence,
            long_labels=labels,
            max_length=self.max_length
        )
        
        # Synthetic exam questions fit in the first chunk.
        encoded = encoded_chunks[0]
        label_ids = processed_labels[0]
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": label_ids
        }

# Compute token-level accuracy and F1 metrics during training.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Exclude ignored CLS, SEP, and padding positions.
    true_labels = []
    true_predictions = []
    for pred_row, label_row in zip(predictions, labels):
        for p, l in zip(pred_row, label_row):
            if l != -100:
                true_labels.append(id_to_label[l])
                true_predictions.append(id_to_label[p])
    
    # Calculate the reported metrics.
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

# Train and save the BIO token classifier.
def train_model():
    # 1. Generate the synthetic BIO dataset.
    print("===== Generating the exam-question BIO dataset =====")
    raw_dataset = generate_dataset()
    print(f"Dataset size: {len(raw_dataset)}")
    print("Example:", raw_dataset[0])

    # 2. Split training and validation data (80/20).
    train_size = int(0.8 * len(raw_dataset))
    train_data = raw_dataset[:train_size]
    val_data = raw_dataset[train_size:]
    
    # 3. Build dataset objects.
    train_dataset = ExamBIODataset(train_data, tokenizer, label_to_id, max_length)
    val_dataset = ExamBIODataset(val_data, tokenizer, label_to_id, max_length)
    
    # 4. Configure training.
    training_args = TrainingArguments(
        output_dir="./exam_bio_model",  # Intermediate checkpoints
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        logging_dir="./exam_bio_logs",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        no_cuda=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_accumulation_steps=4
)

    # 5. Build the Trainer and start training.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("===== Starting training =====")
    trainer.train()
    
    # 6. Save the final model.
    print("===== Training complete; saving the model =====")
    model.save_pretrained("./exam_bio_final_model")
    tokenizer.save_pretrained("./exam_bio_final_model")
    
    # 7. Run a quick inference check.
    test_sentence = "今年英语四级报名截止时间是什么"
    chars, pred_labels = predict_long_text(test_sentence)
    print(f"\nTest sentence: {test_sentence}")
    print("BIO labels:")
    for char, label in zip(chars, pred_labels):
        print(f"{char}\t{label}")

# Command-line entry point
if __name__ == "__main__":
    train_model()

