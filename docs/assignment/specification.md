# Avaliação Parcial 2 (AP2) - Mineração de Dados
## Predição de Gênero em Reviews de E-commerce

---

## 1. Objetivo

Desenvolver um **modelo de classificação** que prediga o gênero do autor (Masculino/Feminino) a partir do texto de reviews de produtos. O trabalho avalia a capacidade de:
- Explorar estratégias diferentes de pré-processamento e representação textual
- Selecionar e justificar algoritmos de classificação
- Analisar criticamente os resultados e limitações do modelo
- Comunicar decisões metodológicas de forma clara

---

## 2. Dataset

### Descrição
- **Fonte:** B2W-Reviews01 - Reviews de e-commerce do Americanas.com
- **Período:** Janeiro a Maio de 2018
- **Total de reviews:** 128.237 (após limpeza)
- **Distribuição de gênero:** 51,6% Masculino (M) / 48,4% Feminino (F)

### Arquivos Fornecidos

| Arquivo | Tamanho | Uso |
|---------|---------|-----|
| `treino.csv` | ~76.942 reviews | Treinamento do modelo |
| `validacao.csv` | ~25.647 reviews | Validação e ajuste de hiperparâmetros |
| `teste.csv` | ~25.648 reviews | **Oculto - Avaliação final (revelado ao final das apresentações)** |

### Colunas Relevantes
- `review_text`: Texto da review (principal feature)
- `reviewer_gender`: Gênero do autor ('M' ou 'F') - **rótulo**
- Outras colunas: metadados (opcional, podem ser usadas se desejado)

### Características dos Dados
- Textos em **português brasileiro**
- Linguagem informal, com erros ortográficos, abreviações e gírias
- Textos de tamanho variável (mediana: 16 tokens)
- Possível desbalanceamento entre classes em subgrupos
- Referências: artigos sobre predição de gênero em português disponíveis no diretório

---

## 3. Requisitos

### 3.1 Desenvolvimento
- **Linguagem:** Python 3.x
- **Ambiente:** Google Colab (com notebook .ipynb)
- **Bibliotecas:** scikit-learn, pandas, numpy, nltk, spacy (ou similares)

### 3.2 Pipeline Obrigatório
Seu modelo deve incluir:

1. **Pré-processamento** (escolha suas técnicas):
   - Limpeza de texto (remoção de URLs, caracteres especiais, etc.)
   - Tokenização
   - Remoção de stopwords (opcional, mas justifique)
   - Normalização (lowercasing, stemming/lemmatização, etc.)

2. **Representação Textual** (escolha uma ou combine):
   - Bag of Words (BoW) / TF-IDF
   - N-gramas (caracteres ou palavras)
   - Word Embeddings (Word2Vec, fastText, GloVe)
   - Outras abordagens criativas

3. **Classificação** (qualquer algoritmo):
   - SVM, Regressão Logística, Naive Bayes
   - Random Forest, Gradient Boosting
   - Redes Neurais (MLP, CNN, LSTM)
   - Ensemble methods
   - Qualquer outra técnica que justifique

4. **Validação e Avaliação**:
   - Divisão treino/validação (80/20 do arquivo treino.csv)
   - Validação cruzada (opcional)
   - Métricas: Acurácia, Precision, Recall, **F1 Score**, Matriz de Confusão
   - **F1 Score** será a métrica de ranqueamento final

### 3.3 Saída: Modelo Serializado
- Salvar o modelo **treinado** em formato `.pkl` (pickle)
- Arquivo deve ser nomeado: `modelo_[seu_nome].pkl`
- O modelo deve estar pronto para fazer predições em dados **brutos** (textos)
- Seu modelo será carregado e avaliado no conjunto de teste oculto

---

## 4. Entrega

### 4.1 Arquivos Obrigatórios

1. **Notebook (`AP2_[seu_nome].ipynb`)**
   - Toda a análise, pré-processamento, treinamento e validação
   - Células bem estruturadas com explicações em markdown
   - Código comentado (não por demais, mas o necessário)
   - Incluir: exploração dos dados, decisões tomadas, resultados

2. **Modelo Serializado (`modelo_[seu_nome].pkl`)**
   - Arquivo pickle contendo o modelo treinado
   - Pronto para fazer predições em novos textos

3. **Slides para Apresentação (`slides_[seu_nome].pdf` ou `.pptx`)**
   - 5-7 slides com:
     * Problema e abordagem
     * Estratégia de pré-processamento e features
     * Algoritmo(s) escolhido(s) e por quê
     * Resultados (métricas no conjunto de validação)
     * Lições aprendidas e limitações
     * Possíveis melhorias futuras

### 4.2 Deadline de Entrega
- **Data:** Segunda-feira, 12 de maio de 2026 até 23:59
- **Local:** Google Classroom (código: hpc4sa66)
- **Estrutura de upload:** Pasta com seu nome contendo: notebook, modelo .pkl e slides

---

## 5. Apresentação e Avaliação

### 5.1 Apresentação em Sala
- **Data:** 13/05/2026 (primeira data) e 18/05/2026 (segunda data)
- **Duração:** Máximo **10 minutos**
- **Formato:** Apresentação seguida de perguntas do professor

### 5.2 Critérios de Avaliação

| Critério | Peso | Descrição |
|----------|------|-----------|
| **F1 Score no Teste Oculto** | 40% | Desempenho do modelo no conjunto de teste revelado ao final |
| **Qualidade Metodológica** | 25% | Coerência entre decisões, justificativa das escolhas, pipeline apropriado |
| **Análise Crítica** | 15% | Discussão de resultados, limitações, vieses, alternativas testadas |
| **Apresentação e Comunicação** | 15% | Clareza na exposição, domínio do conteúdo, organização dos slides |
| **Qualidade do Código** | 5% | Reprodutibilidade, clareza, documentação (bônus se bem feito) |

### 5.3 Ranqueamento Final
Após todas as apresentações, o professor:
1. Carregará cada modelo `.pkl` dos alunos
2. Avaliará no conjunto de teste oculto
3. Calculará F1 Score para cada um
4. **Revelará o ranking público** (1º, 2º, 3º, etc.)

---

## 6. Referências

### Artigos Fornecidos (Leitura Recomendada)
- **Real et al. (2019):** "B2W-Reviews01: An open product reviews corpus"
  - Descreve o dataset e características
  
- **Morais & Merschmann (2021):** "Uma Abordagem Híbrida para Predição de Gênero a partir de Textos em Português"
  - Abordagem com heurística + classificador
  - Baseline: ~89.6% de acurácia no B2W-Reviews01
  
- **Morais & Merschmann (2022):** "A Cascade Approach for Gender Prediction from Texts in Portuguese Language"
  - Abordagem em cascata
  - Resultados: 80%+ de acurácia em múltiplos datasets

### Bibliotecas Úteis
- **Pré-processamento:** NLTK, spaCy, TextBlob
- **Features:** scikit-learn (TfidfVectorizer), gensim (Word2Vec)
- **Classificação:** scikit-learn, XGBoost, LightGBM, TensorFlow/Keras
- **Avaliação:** scikit-learn (metrics)

---

## 7. Exemplo: Como Salvar e Carregar o Modelo

Ver arquivo `template_modelo.py` para código de referência.

**Resumo:**
```python
# Treinar seu pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(...)),
    ('classifier', LogisticRegression(...))
])
model.fit(X_train, y_train)

# Salvar
import pickle
pickle.dump(model, open('modelo_seu_nome.pkl', 'wb'))

# Carregar (isso que o professor fará)
model = pickle.load(open('modelo_seu_nome.pkl', 'rb'))
predicoes = model.predict(['novo texto aqui'])
```

---

## 8. Dúvidas Frequentes

**P: Posso usar dados externos (como dicionários, wordlists)?**  
R: Sim, desde que justifique e cite a fonte.

**P: Posso usar modelos pré-treinados (BERT, etc)?**  
R: Sim, mas tenha em mente que Colab pode ter limitações de GPU.

**P: E se o modelo não converge ou dá erro no teste oculto?**  
R: Será avaliado pelo F1 Score que conseguir. Justifique as limitações na apresentação.

**P: Posso usar ensemble com múltiplos modelos?**  
R: Sim, é uma boa estratégia! Salve o ensemble final em um arquivo .pkl único.

**P: Quanto tempo devo gastar no trabalho?**  
R: Estimado 15-20 horas (exploração, experimentação, refinamento).

---

## 9. Cronograma

| Data | Atividade |
|------|-----------|
| 06/05 (terça) | Aula 21: Laboratório AP2 - Desenvolvimento orientado |
| 11/05 (domingo) | Aula 22: Laboratório AP2 - Refinamento e análise de resultados |
| 12/05 (segunda) 23:59 | **DEADLINE: Entrega de notebook, modelo .pkl e slides** |
| 13/05 (terça) | Aula 23: Seminários AP2 - Apresentações (1ª rodada, ~10 alunos) |
| 18/05 (domingo) | Aula 24: Seminários AP2 - Apresentações (2ª rodada, ~10 alunos) |
| 18/05 (após apresentações) | Revelação do ranking final (F1 no teste oculto) |

---

## 10. Observações Importantes

1. **Reprodutibilidade:** Use `random_state` / `seed` para garantir que seus experimentos sejam reprodutíveis.

2. **Balanceamento:** O dataset é aproximadamente balanceado (51% M / 48% F), mas verifique em subgrupos.

3. **Overfitting:** Monitore a diferença entre F1 no treino e validação. Use técnicas de regularização se necessário.

4. **Dados Brutos:** Seu modelo receberá textos brutos (como estão no dataset). Certifique-se de que o pré-processamento é aplicado internamente (dentro do pipeline).

5. **Viés:** Considere discutir possíveis vieses na predição de gênero (p.ex., características culturais de linguagem).

---

**Boa sorte! 🚀**
