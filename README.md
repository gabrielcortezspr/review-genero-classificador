# Review Author Gender Classifier

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Text classifier that estimates the self-declared gender (M/F) of the author of a Brazilian
e-commerce review. Built for the Data Mining course on the
[B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) dataset (Americanas.com,
~130k reviews).

## Approach

- **Preprocessing:** HTML and URL removal, digits normalized to `NUM`, lowercasing.
- **Representation:** word TF-IDF + character n-gram TF-IDF, plus style features (text
  length, exclamation marks, emojis, product-category words), combined in a custom
  scikit-learn transformer ([`src/text_features.py`](src/text_features.py)).
- **Model:** `LogisticRegression` (`C=0.5`, `class_weight="balanced"`) with sigmoid
  calibration (`CalibratedClassifierCV`), all inside a single `Pipeline` that is pickled
  as one object.

## Validation results

| Metric | Value |
|---|---|
| Weighted F1 | **0.684** |
| Accuracy | 0.685 |
| F1 (F) / F1 (M) | 0.659 / 0.707 |

25,647 validation reviews. Confusion matrix and discussion in
[`docs/slides.pdf`](docs/slides.pdf).

## Limitations

Gender inferred from text is a weak and biased signal: the model mostly learns **which
product category was bought**, not anything about who wrote the review. This is an exercise
in text classification, not a tool for inferring people's characteristics.

## Getting started

```bash
pip install -r requirements.txt
python src/train.py                         # trains on data/train.csv, validates, saves models/
python src/predict.py                       # predicts a few sample reviews
python src/predict.py "Adorei o perfume"    # or your own texts
```

## Project structure

```
├── src/
│   ├── text_features.py   # TextFeatureExtractor: preprocessing + TF-IDF + style features
│   ├── train.py           # training and validation
│   └── predict.py         # loads the saved model and predicts
├── models/                # trained pipeline (gender_classifier.pkl)
├── data/                  # train/validation splits provided by the course
├── notebooks/             # exploration notebook
└── docs/                  # slides, dataset paper and the assignment brief (Portuguese)
```

## License

Code under [MIT](LICENSE). The B2W-Reviews01 data is distributed under its own license
(CC BY-NC-SA 4.0).
