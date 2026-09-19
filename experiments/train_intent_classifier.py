"""Train and evaluate the synthetic TF-IDF intent-classification baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_DATA_PATH = Path(__file__).with_name("intent_data.csv")
DEFAULT_MODEL_PATH = Path(__file__).with_name("intent_model.pkl")
DEFAULT_SEED = 42


def train(data_path: Path, model_path: Path, seed: int) -> None:
    print(f"Loading intent training data: {data_path}")
    frame = pd.read_csv(data_path)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Intent data is missing columns: {sorted(missing_columns)}")

    texts = frame["text"].astype(str)
    labels = frame["label"].astype(str)
    print(f"Loaded {len(frame)} examples")
    print("Intent label distribution:")
    print(labels.value_counts())

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    classifier = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=5000)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )

    print("Training the intent classifier...")
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    print("\nValidation results:")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_path)
    print(f"Saved the intent model to: {model_path}")
    print(f"Seed: {seed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    train(args.data, args.output, args.seed)


if __name__ == "__main__":
    main()
