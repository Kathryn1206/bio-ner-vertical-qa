# Train a lightweight user-intent classifier.

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. File paths
DATA_PATH = "intent_data.csv"
MODEL_PATH = "intent_model.pkl"

# 2. Load the dataset
print("Loading intent training data...")
df = pd.read_csv(DATA_PATH)

assert "text" in df.columns and "label" in df.columns, \
    "intent_data.csv must contain text and label columns"

texts = df["text"].astype(str)
labels = df["label"].astype(str)

print(f"Loaded {len(df)} training examples")
print("Intent label distribution:")
print(labels.value_counts())

# 3. Split training and validation data
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# 4. Build the model pipeline
# Character-aware TF-IDF bigrams and logistic regression provide a strong Chinese baseline.
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

print("Training the intent classifier...")
pipeline.fit(X_train, y_train)

# 5. Evaluate on the validation split
print("\nValidation results:")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

# 6. Save the trained model
joblib.dump(pipeline, MODEL_PATH)
print(f"\nSaved the intent model to: {MODEL_PATH}")

print("\nIntent model training complete.")
print("Next step: load intent_model.pkl in the application pipeline.")
