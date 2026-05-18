# Predição de Gênero em Reviews

**Aluno:** Seu Nome

**Data:** 12/05/2026

---

## Slide 1 — Capa
- Título: Predição de Gênero em Reviews
- Nome: Seu Nome
- Data: 12/05/2026

---

## Slide 2 — Problema
- Tarefa: Classificação binária do gênero do autor da review (M/F).
- Dataset: B2W-Reviews01 (~128k reviews).
- Objetivo: Treinar um classificador que estime `reviewer_gender` a partir do `review_text`.

---

## Slide 3 — Abordagem
- Pré-processamento: limpeza de HTML/URLs, normalização de dígitos para `NUM`, remoção de caracteres especiais, lowercasing.
- Representação: TF-IDF (caracteres + palavras, n-grams), features estilísticas (comprimento, emojis, exclamações, etc.).
- Justificativas:
  - Remover ruído (URLs/HTML) melhora sinais textuais.
  - TF-IDF captura importância de termos em corpus grande; char n-grams ajudam morfologia/terminações relacionadas a gênero.

---

## Slide 4 — Dados & Features
- Tamanho (exemplo): Treino + Validação usados (ver notebook para contagens exatas).
- Exemplos de reviews (amostra): apresentar 2–3 trechos curtos.
- Distribuição de gênero: mostrar counts/freq (imbalance tratado com `class_weight='balanced'`).
- Dimensionalidade das features: TF-IDF resultou em matriz esparsa (ex.: (n_samples, n_features)).

---

## Slide 5 — Modelo
- Algoritmo: `LogisticRegression` com calibrador (`CalibratedClassifierCV`) em pipeline com extrator de features.
- Hiperparâmetros principais: `C=0.5`, `max_iter=10000`, `class_weight='balanced'`.
- Por que esse modelo?
  - Interpretable, eficiente em alta dimensão, convergência estável com TF-IDF.
- Alternativas: SVM, Random Forest, ensembles; possível uso de modelos de linguagem pré-treinados (BERT) como melhoria.

---

## Slide 6 — Resultados (Validação)
Carregando dados...

Treinando...

Validando...

**F1:** 0.6838

**Matriz de Confusão:**
```
[[7819 4595]
 [3488 9745]]
```

**Relatório de Classificação:**
```
              precision    recall  f1-score   support

           F     0.6915    0.6299    0.6592     12414
           M     0.6796    0.7364    0.7069     13233

    accuracy                         0.6848     25647
   macro avg     0.6855    0.6831    0.6830     25647
weighted avg     0.6854    0.6848    0.6838     25647
```

---

## Slide 7 — Análise & Limitações
- Pontos fortes:
  - Resultados razoáveis com pipeline TF-IDF + LogisticRegression.
  - Recursos estilísticos ajudam captura de sinais não-lexicais.
- Desafios:
  - Ruído e ambiguidade em textos curtos.
  - Palavras de categoria (produtos) podem viciar predições.
- Limitações e vieses:
  - Possível viés demográfico no dataset; terminar em generalização limitada.
- Melhorias futuras:
  - Usar `TextFeatureExtractor` do repositório (char+word+style), calibração adicional, fine-tune de modelos pré-treinados, oversampling/undersampling, e tuning de hiperparâmetros.

---

## Slide 8 — Exemplos (opcional)
- Exemplos previstos corretamente: mostrar 2–3 casos com label real vs previsto.
- Exemplos de erros: casos ambíguos e padrões observados (ex.: terminações que sugerem gênero).

---

## Top features (insights)

Top features extraídas do modelo (valores e polaridade):

===================
TOP MASCULINAS
===================
CHAR_sfeito (3.8968)
WORD_satisfeito (3.4192)
WORD_obrigado (1.8465)
CHAR_rigado (1.7784)
CHAR_nado (1.6896)
CHAR_igado (1.6642)
WORD_decepcionado (1.6541)
CHAR_feito (1.5268)
WORD_muito satisfeito (1.4933)
CHAR_onado (1.3277)
CHAR_ top (1.2791)
WORD_estou satisfeito (1.2542)
WORD_top (1.2525)
CHAR_ionado (1.2178)
CHAR_gado (1.0952)
CHAR_nado  (1.0352)
CHAR_eito (1.0175)
CHAR_ feita (0.9660)
WORD_insatisfeito (0.9075)
CHAR_endido (0.8572)
CHAR_ndido (0.8500)
CHAR_ado  (0.8460)
CHAR_ meu p (0.8312)
WORD_satisfeito com (0.8276)
CHAR_ito, p (0.8133)

===================
TOP FEMININAS
===================
CHAR_sfeita (-3.9575)
WORD_satisfeita (-3.1697)
WORD_amei (-2.5661)
CHAR_amei (-2.4087)
WORD_obrigada (-2.0189)
CHAR_rigada (-2.0010)
CHAR_feita (-1.8954)
CHAR_igada (-1.8817)
CHAR_onada (-1.8741)
CHAR_ionada (-1.7997)
WORD_decepcionada (-1.7490)
CHAR_eita (-1.6488)
CHAR_lind (-1.6468)
CHAR_ amei (-1.4769)
CHAR_gada (-1.4673)
WORD_muito satisfeita (-1.3769)
CHAR_nada (-1.3765)
WORD_estou satisfeita (-1.1709)
CHAR_amei  (-1.1353)
CHAR_ada! (-1.1276)
WORD_satisfeita com (-1.1099)
CHAR_lindo (-1.1093)
WORD_insatisfeita (-1.0946)
CHAR_dorei (-1.0756)

---

Modelo salvo: modelo_gabriel_cortez.pkl


dados finais
