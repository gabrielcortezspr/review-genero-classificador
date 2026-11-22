"""Load the trained model and predict the author's gender for a few reviews.

Usage:
    python src/predict.py                       # built-in examples
    python src/predict.py "Produto muito bom"   # your own texts
"""

import pickle
import sys
from pathlib import Path

import text_features  # noqa: F401  (needed to unpickle TextFeatureExtractor)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gender_classifier.pkl"

EXAMPLES = [
    "Produto muito bom, chegou antes do prazo.",
    "Não gostei, a qualidade é ruim.",
    "Comprei para minha filha e ela adorou o kit de maquiagem.",
    "A furadeira é potente e a bateria dura bastante.",
]


def main():
    if not MODEL_PATH.exists():
        sys.exit(f"{MODEL_PATH} not found. Train the model first: python src/train.py")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    texts = sys.argv[1:] or EXAMPLES
    labels = model.predict(texts)
    probs = model.predict_proba(texts)
    classes = list(model.classes_)

    for text, label, p in zip(texts, labels, probs):
        print(f"{label}  (P={p[classes.index(label)]:.2f})  {text}")


if __name__ == "__main__":
    main()
