"""Train the review gender classifier and evaluate it on the validation split.

Usage:
    python src/train.py
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from text_features import TextFeatureExtractor

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "gender_classifier.pkl"


def load_split(name: str):
    df = pd.read_csv(DATA_DIR / f"{name}.csv", low_memory=False)
    return df["review_text"].fillna(""), df["reviewer_gender"]


def build_model() -> Pipeline:
    base_model = LogisticRegression(
        max_iter=10000,
        class_weight="balanced",
        random_state=42,
        C=0.5,
    )
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
    return Pipeline([
        ("features", TextFeatureExtractor()),
        ("classifier", calibrated_model),
    ])


def main():
    print("Loading data...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")

    print("Training...")
    model = build_model()
    model.fit(X_train, y_train)

    print("Validating...")
    preds = model.predict(X_val)
    print(f"\nWeighted F1: {f1_score(y_val, preds, average='weighted'):.4f}")
    print("\nConfusion matrix (rows = true, cols = predicted; order F, M):")
    print(confusion_matrix(y_val, preds, labels=["F", "M"]))
    print("\nClassification report:")
    print(classification_report(y_val, preds, digits=4))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nModel saved to {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
