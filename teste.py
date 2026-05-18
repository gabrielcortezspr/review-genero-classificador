import pickle

# Carregar o modelo
model = pickle.load(open('modelo_gabriel_cortez.pkl', 'rb'))

# Fazer predições em novos textos (do conjunto de teste oculto)
novos_textos = ["Produto muito bom", "Não gostei", "ODEIOOOO", "MACACOS ME MORDam"]
predicoes = model.predict(novos_textos)

print("Predições:", predicoes)  # Deve retornar array com 'M' ou 'F'

# Calcular F1 Score (usando y_true do conjunto de teste)
from sklearn.metrics import f1_score
y_true = ['M', 'F']  # Valores verdadeiros do teste oculto
f1 = f1_score(y_true, predicoes, average='weighted')
print(f"F1 Score: {f1:.4f}")
