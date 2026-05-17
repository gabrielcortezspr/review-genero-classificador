"""
SVM_Modelo.py

Modelo SVM para predicao de genero usando apenas:
- review_title
- review_text
- reviewer_gender

Sem limpeza (limpeza zero). Apenas preenche nulos com string vazia.
"""

import pickle
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC


def build_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Cria campos de texto titulo, review e ambos, mantendo limpeza zero."""
    df = df.copy()
    df["review_title"] = df["review_title"].fillna("")
    df["review_text"] = df["review_text"].fillna("")
    df["titulo"] = df["review_title"].astype(str)
    df["review"] = df["review_text"].astype(str)
    df["ambos"] = (df["review_title"].astype(str) + " " + df["review_text"].astype(str)).str.strip()
    return df


def make_vectorizer() -> FeatureUnion:
    """Cria um FeatureUnion de palavras e caracteres (sem limpeza)."""
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=False,
    )
    return FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])


def select_column(X: pd.DataFrame, col: str) -> np.ndarray:
    """Seleciona uma coluna de texto como array de strings."""
    return X[col].astype(str).values


def compute_style_features(texts: np.ndarray) -> np.ndarray:
    """Extrai features estilisticas sem limpeza.

    Features:
    - comprimento do texto
    - contagem de espacos duplos e taxa por caractere
    - pontuacao colada vs espacada
    - espaco depois de virgula
    - pontuacao final (.,!,?)
    - excessos de pontuacao ("!!", "??", "...")
    - alongamento de letras (max repeticao, contagem de seq > 2)
    - proporcao de maiusculas
    - risadas (kkk, rsrs, haha)
    - emoticons simples
    - tamanho: numero de tokens, media de tamanho de palavra
    - repeticao de palavra (tokens repetidos consecutivos)
    - proporcao de acentos
    - proporcao de digitos
    """
    feats = []
    accent_re = re.compile(r"[\u00C0-\u017F]")
    for text in texts:
        s = "" if text is None else str(text)
        length = len(s)
        letters = re.findall(r"[A-Za-z\u00C0-\u017F]", s)
        letters_count = len(letters)
        upper_count = sum(1 for ch in letters if ch.isupper())
        digits_count = len(re.findall(r"\d", s))

        double_spaces = len(re.findall(r" {2,}", s))
        double_space_ratio = double_spaces / max(1, length)

        comma_tight = len(re.findall(r",\S", s))
        dot_tight = len(re.findall(r"\.\S", s))
        space_before_punct = len(re.findall(r"\s[,\.]", s))
        comma_space_after = len(re.findall(r",\s+[A-Za-z\u00C0-\u017F]", s))
        comma_count = s.count(",")

        ends_with_punct = 1.0 if re.search(r"[.!?]$", s.strip()) else 0.0
        exclam_double = len(re.findall(r"!!", s))
        quest_double = len(re.findall(r"\?\?", s))
        ellipsis = len(re.findall(r"\.\.\.", s))

        max_repeat = 0
        repeat_seq = 0
        for m in re.finditer(r"([A-Za-z\u00C0-\u017F])\1+", s):
            run_len = len(m.group(0))
            max_repeat = max(max_repeat, run_len)
            if run_len >= 3:
                repeat_seq += 1

        laugh_k = len(re.findall(r"k{3,}", s, flags=re.IGNORECASE))
        laugh_rs = len(re.findall(r"r{1,}s{1,}r{1,}s{1,}", s, flags=re.IGNORECASE))
        laugh_ha = len(re.findall(r"ha{2,}", s, flags=re.IGNORECASE))

        emoticons = len(re.findall(r"(:\)|:\(|;\-\)|;\))", s))

        tokens = re.findall(r"\S+", s)
        token_count = len(tokens)
        avg_word_len = (sum(len(t) for t in tokens) / max(1, token_count))
        repeated_tokens = sum(1 for i in range(1, token_count) if tokens[i].lower() == tokens[i - 1].lower())

        accent_count = len(accent_re.findall(s))
        accent_ratio = accent_count / max(1, letters_count)
        digit_ratio = digits_count / max(1, length)

        comma_space_ratio = comma_space_after / max(1, comma_count)
        tight_punct = comma_tight + dot_tight

        feats.append([
            length,
            double_spaces,
            double_space_ratio,
            tight_punct,
            space_before_punct,
            comma_space_ratio,
            ends_with_punct,
            exclam_double,
            quest_double,
            ellipsis,
            max_repeat,
            repeat_seq,
            upper_count / max(1, letters_count),
            laugh_k,
            laugh_rs,
            laugh_ha,
            emoticons,
            token_count,
            avg_word_len,
            repeated_tokens,
            accent_ratio,
            digit_ratio,
        ])
    return np.asarray(feats, dtype=float)


def make_model_pipeline() -> Pipeline:
    """Pipeline final com tres ramos (titulo, review, ambos) + LinearSVC."""
    titulo_text = Pipeline([
        ("select", FunctionTransformer(select_column, kw_args={"col": "titulo"})),
        ("tfidf", make_vectorizer()),
    ])
    review_text = Pipeline([
        ("select", FunctionTransformer(select_column, kw_args={"col": "review"})),
        ("tfidf", make_vectorizer()),
    ])
    ambos_text = Pipeline([
        ("select", FunctionTransformer(select_column, kw_args={"col": "ambos"})),
        ("tfidf", make_vectorizer()),
    ])

    titulo_style = Pipeline([
        ("select", FunctionTransformer(select_column, kw_args={"col": "titulo"})),
        ("style", FunctionTransformer(compute_style_features)),
        ("scale", StandardScaler(with_mean=False)),
    ])
    review_style = Pipeline([
        ("select", FunctionTransformer(select_column, kw_args={"col": "review"})),
        ("style", FunctionTransformer(compute_style_features)),
        ("scale", StandardScaler(with_mean=False)),
    ])
    ambos_style = Pipeline([
        ("select", FunctionTransformer(select_column, kw_args={"col": "ambos"})),
        ("style", FunctionTransformer(compute_style_features)),
        ("scale", StandardScaler(with_mean=False)),
    ])

    # FeatureUnion combina os tres blocos
    all_features = FeatureUnion([
        ("titulo_text", titulo_text),
        ("review_text", review_text),
        ("ambos_text", ambos_text),
        ("titulo_style", titulo_style),
        ("review_style", review_style),
        ("ambos_style", ambos_style),
    ])

    model = LinearSVC(max_iter=5000)

    return Pipeline([
        ("features", all_features),
        ("classifier", model),
    ])


class ModeloGeneroSVM:
    """Wrapper para aceitar lista de textos no predict()."""

    def __init__(self) -> None:
        self.pipeline = make_model_pipeline()

    def _to_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            texts = list(X)
            df = pd.DataFrame({
                "review_title": ["" for _ in texts],
                "review_text": texts,
            })
        if "titulo" not in df.columns or "review" not in df.columns or "ambos" not in df.columns:
            df = build_text_fields(df)
        return df

    def fit(self, X, y):
        df = self._to_dataframe(X)
        self.pipeline.fit(df[["titulo", "review", "ambos"]], y)
        return self

    def predict(self, X):
        df = self._to_dataframe(X)
        return self.pipeline.predict(df[["titulo", "review", "ambos"]])


def main() -> None:
    # Carregar dados
    train_df = pd.read_csv("treino.csv")
    val_df = pd.read_csv("validacao.csv")

    # Selecionar apenas as colunas necessarias
    keep_cols = ["review_title", "review_text", "reviewer_gender"]
    train_df = train_df[keep_cols]
    val_df = val_df[keep_cols]

    train_df = build_text_fields(train_df)
    val_df = build_text_fields(val_df)

    X_train = train_df[["titulo", "review", "ambos"]]
    y_train = train_df["reviewer_gender"].values

    X_val = val_df[["titulo", "review", "ambos"]]
    y_val = val_df["reviewer_gender"].values

    # Treinar
    print("Treinando SVM...")
    modelo = ModeloGeneroSVM()
    modelo.fit(X_train, y_train)

    # Validar
    y_pred = modelo.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="weighted")
    print(f"F1 (validacao): {f1:.4f}")

    print("\nMatriz de confusao:")
    print(confusion_matrix(y_val, y_pred, labels=["M", "F"]))

    print("\nRelatorio:")
    print(classification_report(y_val, y_pred, digits=4))

    # Teste rapido
    exemplos = [
        "Produto excelente, adorei.",
        "Nao gostei do atendimento, pessimo.",
    ]
    print("\nExemplos:")
    print(modelo.predict(exemplos))

    # Salvar modelo
    model_path = "modelo_gabriel_cortez.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\nModelo salvo em: {model_path}")


if __name__ == "__main__":
    main()
