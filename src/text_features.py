"""Feature extraction for the review gender classifier.

`TextFeatureExtractor` is a scikit-learn transformer that combines word TF-IDF,
character n-gram TF-IDF and hand-crafted style features. It lives in its own module
so the pickled pipeline can be loaded by `predict.py`.
"""

import re

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# ============================================================
# PALAVRAS DE CATEGORIA
# ============================================================

CATEGORY_WORDS = {

    "shampoo",
    "condicionador",
    "hidratante",
    "perfume",
    "cabelo",
    "cabelos",
    "maquiagem",
    "serum",
    "máscara",
    "mascara",
    "escova",

    "monitor",
    "fps",
    "gpu",
    "processador",
    "ssd",
    "placa",
    "renderizacao",
    "renderização",
    "notebook",
    "pc",
    "roteador",
    "televisor",
    "equipamento",

    "jogo",
    "jogos",
    "game",
    "games",
    "steam",
}


# ============================================================
# RELACOES SOCIAIS
# ============================================================

RELATIONSHIP_WORDS = {

    "esposa",
    "esposo",
    "marido",
    "namorada",
    "namorado",
    "mulher",
    "noiva",
    "noivo",
    "amiga",
    "amigo",
    "mãe",
    "mae",
    "pai",
    "filha",
    "filho",
}


# ============================================================
# NORMALIZACAO
# ============================================================

def remove_words(text: str) -> str:

    blocked_words = (
        CATEGORY_WORDS
        | RELATIONSHIP_WORDS
    )

    tokens = text.split()

    filtered = []

    for token in tokens:

        clean_token = re.sub(
            r"[^\wÀ-ÿ]",
            "",
            token.lower()
        )

        if clean_token not in blocked_words:

            filtered.append(token)

    return " ".join(filtered)


def normalize_text(text: str) -> str:

    text = "" if text is None else str(text)

    text = re.sub(
        r"http\S+|www\S+",
        " URL ",
        text
    )

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"\d+",
        " NUM ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = remove_words(text)

    return text.strip()


# ============================================================
# FEATURES ESTILISTICAS
# ============================================================

def compute_style_features(texts):

    feats = []

    accent_re = re.compile(r"[\u00C0-\u017F]")

    emojis_re = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE
    )

    for text in texts:

        s = "" if text is None else str(text)

        length = len(s)

        tokens = re.findall(r"\S+", s)

        token_count = len(tokens)

        unique_tokens = len(
            set(t.lower() for t in tokens)
        )

        letters = re.findall(
            r"[A-Za-z\u00C0-\u017F]",
            s
        )

        letters_count = len(letters)

        upper_count = sum(
            1 for ch in letters
            if ch.isupper()
        )

        digits_count = len(
            re.findall(r"\d", s)
        )

        exclamations = s.count("!")

        questions = s.count("?")

        elongated_words = len(
            re.findall(
                r"([a-zA-Z])\1{2,}",
                s
            )
        )

        laugh_k = len(
            re.findall(
                r"k{3,}",
                s,
                flags=re.IGNORECASE
            )
        )

        laugh_ha = len(
            re.findall(
                r"(ha){2,}",
                s,
                flags=re.IGNORECASE
            )
        )

        laugh_rs = len(
            re.findall(
                r"(rs){2,}",
                s,
                flags=re.IGNORECASE
            )
        )

        emojis_count = len(
            emojis_re.findall(s)
        )

        diminutives = len(
            re.findall(
                r"\b\w+(inho|inha|zinho|zinha)\b",
                s.lower()
            )
        )

        intensifiers = len(
            re.findall(
                r"\b(muito|super|mega|extremamente|demais)\b",
                s.lower()
            )
        )

        first_person = len(
            re.findall(
                r"\b(eu|meu|minha|pra mim|comprei)\b",
                s.lower()
            )
        )

        accent_count = len(
            accent_re.findall(s)
        )

        avg_word_len = (
            sum(len(t) for t in tokens)
            / max(1, token_count)
        )

        type_token_ratio = (
            unique_tokens
            / max(1, token_count)
        )

        feats.append([

            length,
            token_count,
            unique_tokens,
            avg_word_len,
            type_token_ratio,
            upper_count / max(1, letters_count),
            digits_count / max(1, length),
            exclamations,
            questions,
            elongated_words,
            laugh_k,
            laugh_ha,
            laugh_rs,
            emojis_count,
            diminutives,
            intensifiers,
            first_person,
            accent_count / max(1, letters_count),
        ])

    return np.asarray(
        feats,
        dtype=float
    )


# ============================================================
# TRANSFORMER SKLEARN
# ============================================================

class TextFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            min_df=10,
            max_df=0.80,    
            sublinear_tf=True,
            max_features=10000,
        )

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(4, 6),
            lowercase=True,
            min_df=5,
            sublinear_tf=True,
            max_features=120000,
        )

        self.scaler = StandardScaler(
            with_mean=False
        )

    def fit(self, X, y=None):

        texts = pd.Series(X).astype(str)

        texts = texts.map(normalize_text)

        self.word_vectorizer.fit(texts)

        self.char_vectorizer.fit(texts)

        style_features = compute_style_features(
            texts.values
        )

        self.scaler.fit(style_features)

        return self

    def transform(self, X):

        texts = pd.Series(X).astype(str)

        texts = texts.map(normalize_text)

        word_features = (
            self.word_vectorizer
            .transform(texts)
        )

        char_features = (
            self.char_vectorizer
            .transform(texts)
        )

        style_features = compute_style_features(
            texts.values
        )

        style_features = (
            self.scaler
            .transform(style_features)
        )

        style_features = csr_matrix(
            style_features
        )

        word_features = word_features * 0.7
        char_features = char_features * 2.5
        style_features = style_features * 4

        return hstack([
            word_features,
            char_features,
            style_features,
        ])

