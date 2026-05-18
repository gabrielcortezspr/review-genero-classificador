# 📦 Instruções de Entrega - AP2

## Resumo Rápido

**O que entregar?**
1. Notebook (`.ipynb`)
2. Modelo treinado (`.pkl`)
3. Slides (`.pdf` ou `.pptx`)

**Quando?**
- **Deadline:** 12 de maio de 2026, até 23:59

**Onde?**
- Google Classroom (código: hpc4sa66)

---

## 📋 Checklist Antes de Entregar

- [ ] Notebook está em Python 3.x
- [ ] Todas as células rodam sem erro (Ctrl+A → Ctrl+Enter)
- [ ] Explicações em markdown estão claras
- [ ] Modelo foi treinado e salvou em `.pkl`
- [ ] Testei carregar o modelo: `pickle.load(open('modelo.pkl', 'rb'))`
- [ ] Modelo retorna predições em M/F para textos brutos
- [ ] F1 Score foi calculado no conjunto de validação
- [ ] Slides têm 5-7 slides (máximo 10 minutos)
- [ ] Nomeei os arquivos corretamente (ver abaixo)

---

## 📁 Estrutura de Entrega

Crie uma **pasta com seu nome** no Google Classroom contendo:

```
seu_nome/
├── AP2_seu_nome.ipynb          # Notebook com toda a análise
├── modelo_seu_nome.pkl         # Modelo treinado (arquivo binary)
└── slides_seu_nome.pdf         # Slides da apresentação (ou .pptx)
```

**Exemplo:**
```
alice_silva/
├── AP2_alice_silva.ipynb
├── modelo_alice_silva.pkl
└── slides_alice_silva.pdf
```

**⚠️ IMPORTANTE:** Use seu sobrenome completo ou o nome que aparece no Classroom.

---

## 📝 Conteúdo do Notebook

O arquivo `.ipynb` deve incluir:

### 1. **Introdução** (1-2 células)
```markdown
# Predição de Gênero - Análise de Reviews

**Aluno:** Seu Nome  
**Data:** 12/05/2026
**Objetivo:** Desenvolver um modelo de classificação para predizer gênero...
```

### 2. **Exploração dos Dados** (3-5 células)
```python
import pandas as pd

# Carregar dados
df_train = pd.read_csv('treino.csv')
df_val = pd.read_csv('validacao.csv')

# Exploração
print(f"Treino: {len(df_train)} reviews")
print(f"Distribuição de gênero:\n{df_train['reviewer_gender'].value_counts()}")
# ... mais análise exploratória
```

### 3. **Pré-processamento** (2-3 células)
```python
# Limpeza, normalização, etc.
def preprocessar(texto):
    # Seu código aqui
    return texto_processado

X_train_clean = df_train['review_text'].apply(preprocessar)
```

### 4. **Representação Textual** (1-2 células)
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(...)
X_train_vec = vectorizer.fit_transform(X_train_clean)
```

### 5. **Treinamento** (1-2 células)
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(...)
model.fit(X_train_vec, y_train)
```

### 6. **Validação e Métricas** (2-3 células)
```python
from sklearn.metrics import f1_score, confusion_matrix

y_pred = model.predict(X_val_vec)
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"F1 Score: {f1:.4f}")
```

### 7. **Salvar Modelo** (1 célula)
```python
import pickle
pickle.dump(model, open('modelo_seu_nome.pkl', 'wb'))
print("✓ Modelo salvo!")
```

### 8. **Conclusões** (1-2 células)
```markdown
## Conclusões

- Melhor F1 Score alcançado: 0.XXXX
- Principais desafios: ...
- Possíveis melhorias: ...
```

---

## 🎯 Conteúdo dos Slides

**Máximo 10 minutos = máximo 7-8 slides**

### Slide 1: Capa
- Título: "Predição de Gênero em Reviews"
- Seu nome
- Data

### Slide 2: Problema
- O que é a tarefa?
- Dataset: B2W-Reviews01 (128k reviews)
- Objetivo: Classificação binária (M/F)

### Slide 3: Abordagem
- Estratégia de pré-processamento
  - Limpeza, tokenização, etc.
- Escolhas justificadas
  - "Por que remover stopwords?" "Por que TF-IDF?"

### Slide 4: Dados & Features
- Tamanho do dataset (treino/validação)
- Exemplos de reviews
- Distribuição de gênero
- Dimensionalidade das features

### Slide 5: Modelo
- Algoritmo escolhido
- Hiperparâmetros principais
- Por que escolheu esse modelo?
- Alternativas testadas?

### Slide 6: Resultados
- **Métricas no validação:**
  - F1 Score
  - Acurácia
  - Precision / Recall
- **Matriz de Confusão**
  - Acerta mais qual classe?

### Slide 7: Análise & Limitações
- O que funcionou bem?
- O que foi desafiador?
- Limitações do modelo
- Viés potencial?
- Melhorias futuras?

### Slide 8 (opcional): Exemplo
- Exemplos de textos preditos corretamente
- Exemplos de erros
- Padrões observados

---

## ✅ Validação Final

Antes de clicar em "Enviar", teste seu modelo:

```python
import pickle

# Carregar modelo
model = pickle.load(open('modelo_seu_nome.pkl', 'rb'))

# Teste com alguns textos
textos_teste = [
    "Este produto é excelente, adorei!",
    "Péssima qualidade, não recomendo.",
    "Chegou rápido, muito bom mesmo!"
]

predicoes = model.predict(textos_teste)
print(predicoes)  # Deve retornar array de M e/ou F
```

Se funcionar, está pronto para entregar!

---

## 📤 Como Enviar no Google Classroom

1. **Criar pasta com seu nome**
   - Clique em "Arquivo" → "Nova pasta"
   - Nomeie: `seu_nome` ou `seu_sobrenome`

2. **Upload dos arquivos**
   - Clique em "Anexar" (clip de papel)
   - Selecione os 3 arquivos:
     - `AP2_seu_nome.ipynb`
     - `modelo_seu_nome.pkl`
     - `slides_seu_nome.pdf`
   - Upload

3. **Enviar**
   - Clique em "Entregar"
   - Confirme a entrega

4. **Confirmar recebimento**
   - Você verá "Entregue" ao lado da tarefa
   - Professor receberá notificação

---

## ⚠️ Erros Comuns

| Problema | Solução |
|----------|---------|
| Arquivo `.pkl` muito grande (>500MB) | Você salvou dados de treino! Salve só o modelo treinado |
| Erro ao carregar `.pkl` | Use Python 3.x e `import pickle` ao salvar e carregar |
| Modelo retorna tipo errado | Verify que `predict()` retorna array de strings ('M' ou 'F') |
| Notebook com erros | Rode todas as células (Ctrl+A → Ctrl+Enter) antes de enviar |
| Deadline passou | Tarde demais, será considerado entrega atrasada |

---

## 💡 Dicas para Melhor Desempenho

1. **Pré-processamento:**
   - Teste diferentes técnicas (stemming vs lemmatização)
   - Monitore F1 Score com/sem cada passo

2. **Features:**
   - Compare TF-IDF vs Word2Vec
   - Experimente n-gramas (1-2, 2-3, etc.)

3. **Modelos:**
   - Teste múltiplos (Logistic, SVM, Random Forest, etc.)
   - Use validação cruzada para tuning

4. **Regularização:**
   - Cuidado com overfitting
   - Se validação >> treino, está overfitting
   - Use L1/L2, dropout, ou reduza features

5. **Análise:**
   - Olhe a matriz de confusão
   - Qual classe é mais fácil de prever?
   - Há padrões nos erros?

6. **Justificação:**
   - Durante apresentação, seja capaz de explicar **por quê**
   - "Por que escolheu TF-IDF?" "Como escolheu hiperparâmetros?"

---

## 📞 Dúvidas Frequentes

**P: Posso usar Google Colab?**  
R: Sim! Exporte o notebook como `.ipynb` e envie no Classroom.

**P: Posso usar bibliotecas extras (BERT, etc)?**  
R: Sim, desde que cite a fonte e justifique.

**P: E se meu modelo não rodar em outro computador?**  
R: Use `random_state` em tudo e documente dependências. Teste com `pickle.load()` em outro lugar.

**P: Quanto tempo devo gastar?**  
R: Estimado 15-20 horas (exploração, testes, refinamento).

**P: Posso enviar depois do deadline?**  
R: Não, será considerado atrasado conforme regras da universidade.

**P: E se o Google Classroom cair?**  
R: Salve um backup local. Contate o professor se houver problema.

---

## 📚 Referências Importantes

Arquivos disponíveis no diretório do trabalho:

1. **ESPECIFICACAO_AP2.md** - Leia completamente antes de começar
2. **template_modelo.py** - Exemplos de código
3. Artigos PDF - Background e baseline

---

## 🎬 Resumo: Passo a Passo

```
1. Carregar dados (treino.csv, validacao.csv)
   ↓
2. Explorar dados (distribuição, exemplos)
   ↓
3. Pré-processar texto (limpeza, normalização)
   ↓
4. Extrair features (TF-IDF, Word2Vec, etc.)
   ↓
5. Treinar modelo (escolha seu algoritmo)
   ↓
6. Validar e avaliar (F1 Score, matriz confusão)
   ↓
7. Salvar modelo em .pkl
   ↓
8. Criar slides com explicação
   ↓
9. Entregar no Classroom antes de 12/05 23:59
   ↓
10. Apresentar em 13/05 ou 18/05 (máximo 10 min)
```

---

**Boa sorte! 🚀**

*Dúvidas? Revise a ESPECIFICACAO_AP2.md ou pergunte ao professor na aula.*
