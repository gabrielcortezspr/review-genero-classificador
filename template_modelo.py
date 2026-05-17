"""
TEMPLATE: Como Salvar e Carregar o Modelo
==========================================

Este arquivo mostra como estruturar seu modelo de forma que ele funcione
com o código de avaliação do professor.

O modelo DEVE:
1. Ser um objeto Python que pode ser serializado com pickle
2. Ter um método predict(X) que aceita lista/array de textos
3. Retornar um array de predições ('M' ou 'F')

Existem duas abordagens recomendadas:
"""

# ============================================================================
# OPÇÃO 1: Usar sklearn Pipeline (RECOMENDADO - Mais Simples)
# ============================================================================

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Carregar dados
df_train = pd.read_csv('treino.csv')
df_val = pd.read_csv('validacao.csv')

X_train = df_train['review_text'].values
y_train = df_train['reviewer_gender'].values

# Criar pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        stop_words=None,  # ou 'portuguese' se usar NLTK
        ngram_range=(1, 2),
        lowercase=True
    )),
    ('classifier', LogisticRegression(
        max_iter=200,
        random_state=42,
        class_weight='balanced'  # útil se houver desbalanceamento
    ))
])

# Treinar
print("Treinando modelo...")
model.fit(X_train, y_train)

# Validar
score = model.score(df_val['review_text'].values, df_val['reviewer_gender'].values)
print(f"Acurácia no conjunto de validação: {score:.4f}")

# Fazer predições em novos textos
novos_textos = [
    "Este produto é excelente, adorei!",
    "Péssima qualidade, não recomendo."
]
predicoes = model.predict(novos_textos)
print(f"\nExemplo de predições:")
for texto, pred in zip(novos_textos, predicoes):
    print(f"  '{texto}' -> {pred}")

# SALVAR o modelo
model_path = 'modelo_seu_nome.pkl'
pickle.dump(model, open(model_path, 'wb'))
print(f"\n✓ Modelo salvo em: {model_path}")


# ============================================================================
# OPÇÃO 2: Criar uma Classe Customizada
# ============================================================================

"""
Se você quiser processar dados de forma mais sofisticada (ex: limpeza especial,
múltiplas features), crie uma classe que envolver o modelo:
"""

import pickle
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

class ModeloGenero:
    """Modelo customizado para predição de gênero"""

    def __init__(self):
        self.pipeline = None
        self.custom_config = {
            'max_features': 3000,
            'ngram_range': (1, 3),
        }

    def _limpar_texto(self, texto):
        """Pré-processamento customizado"""
        # Remover URLs
        texto = re.sub(r'http\S+', '', texto)
        # Remover caracteres especiais (mantém letras, números, espaços)
        texto = re.sub(r'[^a-zA-Zá-ú0-9\s]', '', texto)
        # Lowercase
        texto = texto.lower()
        # Remover espaços múltiplos
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto

    def fit(self, X_train, y_train):
        """Treinar o modelo"""
        # Aplicar limpeza
        X_clean = [self._limpar_texto(texto) for texto in X_train]

        # Criar pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.custom_config['max_features'],
                ngram_range=self.custom_config['ngram_range'],
                lowercase=True
            )),
            ('classifier', SVC(kernel='linear', probability=True, random_state=42))
        ])

        # Treinar
        self.pipeline.fit(X_clean, y_train)
        return self

    def predict(self, X):
        """Fazer predições em novos textos"""
        # Aplicar mesma limpeza
        X_clean = [self._limpar_texto(texto) for texto in X]
        return self.pipeline.predict(X_clean)

    def score(self, X, y):
        """Calcular acurácia"""
        return (self.predict(X) == y).mean()


# Usar a classe customizada
df_train = pd.read_csv('treino.csv')
df_val = pd.read_csv('validacao.csv')

X_train = df_train['review_text'].values
y_train = df_train['reviewer_gender'].values
X_val = df_val['review_text'].values
y_val = df_val['reviewer_gender'].values

# Treinar
print("Treinando modelo customizado...")
modelo = ModeloGenero()
modelo.fit(X_train, y_train)

# Validar
score = modelo.score(X_val, y_val)
print(f"Acurácia no conjunto de validação: {score:.4f}")

# Fazer predições
novos_textos = [
    "Este produto é excelente, adorei!",
    "Péssima qualidade, não recomendo."
]
predicoes = modelo.predict(novos_textos)
print(f"\nExemplo de predições:")
for texto, pred in zip(novos_textos, predicoes):
    print(f"  '{texto}' -> {pred}")

# SALVAR a classe + modelo
model_path = 'modelo_seu_nome.pkl'
pickle.dump(modelo, open(model_path, 'wb'))
print(f"\n✓ Modelo customizado salvo em: {model_path}")


# ============================================================================
# CARREGAR E USAR O MODELO SALVO (Isto é o que o professor fará)
# ============================================================================

"""
Depois de salvar, o professor carregará seu modelo assim:
"""

import pickle

# Carregar o modelo
model = pickle.load(open('modelo_seu_nome.pkl', 'rb'))

# Fazer predições em novos textos (do conjunto de teste oculto)
novos_textos = ["Produto muito bom", "Não gostei"]
predicoes = model.predict(novos_textos)

print("Predições:", predicoes)  # Deve retornar array com 'M' ou 'F'

# Calcular F1 Score (usando y_true do conjunto de teste)
from sklearn.metrics import f1_score
y_true = ['M', 'F']  # Valores verdadeiros do teste oculto
f1 = f1_score(y_true, predicoes, average='weighted')
print(f"F1 Score: {f1:.4f}")


# ============================================================================
# CHECKLIST ANTES DE ENTREGAR
# ============================================================================

"""
Antes de enviar seu modelo, verifique:

[ ] Modelo é um arquivo .pkl
[ ] Nome do arquivo: modelo_[seu_nome].pkl
[ ] Ao carregar com pickle.load(), o objeto funciona
[ ] model.predict(lista_de_textos) retorna array de 'M' ou 'F'
[ ] Modelo foi treinado no arquivo treino.csv
[ ] F1 Score foi calculado corretamente no validacao.csv
[ ] Código no notebook está claro e comentado
[ ] Slides com explicação das escolhas estão prontos

"""
