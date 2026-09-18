from transformers import pipeline
import re
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ['考试报名','考试成绩','考试排名','考试类型','考试时间','考试地点','考试大纲','考试真题']
user_input = input("请输入：")
result = classifier(user_input, candidate_labels)
print(f'User input: {user_input}')
print(f'Predicted intent: {result["labels"][0]}')
print(f'Intent confidence: {result["scores"][0]}')

