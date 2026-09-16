# train_intent.py
# ===============================
# 训练一个「用户意图识别（Intent）」模型
# ===============================

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ========= 1️⃣ 配置路径 =========
DATA_PATH = "intent_data.csv"
MODEL_PATH = "intent_model.pkl"

# ========= 2️⃣ 读取数据 =========
print("📂 正在加载 Intent 训练数据...")
df = pd.read_csv(DATA_PATH)

assert "text" in df.columns and "label" in df.columns, \
    "intent_data.csv 必须包含 text 和 label 两列"

texts = df["text"].astype(str)
labels = df["label"].astype(str)

print(f"✅ 共加载 {len(df)} 条训练样本")
print("Intent 标签分布：")
print(labels.value_counts())

# ========= 3️⃣ 划分训练 / 验证集 =========
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ========= 4️⃣ 构建模型流水线 =========
# 中文场景：TF-IDF + 逻辑回归，非常稳
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

print("🧠 开始训练 Intent 模型...")
pipeline.fit(X_train, y_train)

# ========= 5️⃣ 验证效果 =========
print("\n📊 验证集效果：")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

# ========= 6️⃣ 保存模型 =========
joblib.dump(pipeline, MODEL_PATH)
print(f"\n💾 Intent 模型已保存至：{MODEL_PATH}")

print("\n🎉 Intent 模型训练完成！")
print("👉 下一步：在 main.py 中加载 intent_model.pkl 使用")
