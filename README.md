# Classificador de gênero em reviews

Modelo que estima o gênero declarado do autor (M/F) a partir do texto de uma review de
e-commerce. Trabalho da AP2 de Mineração de Dados, sobre o dataset
[B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) (Americanas.com, 2018).

## Abordagem

- **Pré-processamento:** remoção de HTML e URLs, dígitos normalizados para `NUM`, caixa baixa.
- **Representação:** TF-IDF de palavras e de n-gramas de caracteres, mais features de estilo
  (tamanho do texto, exclamações, emojis, palavras de categoria de produto).
- **Modelo:** `LogisticRegression` (`C=0.5`, `class_weight="balanced"`) calibrada com
  `CalibratedClassifierCV`, tudo num `Pipeline` do scikit-learn.

## Resultado na validação

| Métrica | Valor |
|---|---|
| F1 ponderado | **0,684** |
| Acurácia | 0,685 |
| F1 (F) / F1 (M) | 0,659 / 0,707 |

25.647 reviews de validação. Matriz de confusão e discussão das limitações em
[`slide.md`](slide.md) e [`Documentacao/slides.pdf`](Documentacao/slides.pdf).

## Como rodar

```sh
pip install pandas numpy scipy scikit-learn
python modelo_gabriel_cortez.py   # treina com treino.csv, valida com validacao.csv e salva o .pkl
python teste.py                   # carrega o modelo salvo e prediz textos de exemplo
```

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `modelo_gabriel_cortez.py` | Extração de features, treino e validação |
| `modelo_gabriel_cortez.pkl` | Modelo treinado |
| `notebook.ipynb` | Notebook da entrega |
| `treino.csv`, `validacao.csv` | Partições do dataset fornecidas pela disciplina |
| `Documentacao/` | Especificação do trabalho, instruções e slides |

## Limitações

Gênero inferido de texto é um sinal fraco e enviesado: o modelo aprende sobretudo a categoria
do produto comprado, não algo sobre quem escreve. Serve como exercício de classificação de
texto, não como ferramenta para inferir características de pessoas.

## Licença

[MIT](LICENSE)
