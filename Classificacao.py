import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Importe joblib aqui

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report)

# 1. Carregar os dados
# Certifique-se de que "crop_yield.csv" está no mesmo diretório deste script
df = pd.read_csv("crop_yield.csv").head(100000)

# 2. Codificar variáveis categóricas (exceto a variável alvo 'Fertilizante')
label_encoders_classificacao = {} # Renomeado para evitar confusão com outros encoders no Flask

# Identificar colunas categóricas no DataFrame X
# EXATAMENTE AQUI É A CORREÇÃO: Removido 'axis=0'
categorical_cols_for_classification_model = df.select_dtypes(include=['object']).columns.drop('Fertilizante', errors='ignore')

for col in categorical_cols_for_classification_model:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders_classificacao[col] = le

# 3. Converter variável alvo (Fertilizante) para int
df['Fertilizante'] = df['Fertilizante'].astype(int)

# 4. Separar variáveis independentes e alvo
X = df.drop(columns=['Fertilizante'])
y = df['Fertilizante']

feature_cols_classification = X.columns.tolist() # Salvar as colunas de X

# 5. Padronizar os dados
scaler_classificacao = StandardScaler() # Renomeado para clareza
X_scaled = scaler_classificacao.fit_transform(X)

# 6. Aplicar PCA via SVD manualmente
N_COMPONENTS = 10 # número de componentes principais

# Centralizar os dados (já estão padronizados, mas subtrair a média é bom para o SVD)
X_mean = np.mean(X_scaled, axis=0)
X_centered = X_scaled - X_mean

# Aplicar SVD
U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

# Projeção dos dados nas componentes principais
X_pca = np.dot(U[:, :N_COMPONENTS], np.diag(S[:N_COMPONENTS]))

# Salvar os componentes principais (VT) e a média (X_mean)
# 'VT' são os autovetores, que são os componentes principais (transpostos)
pca_components = VT[:N_COMPONENTS, :]
pca_mean = X_mean

# 7. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y)

# 8. Treinar Regressão Logística
start = time.time()
logreg = LogisticRegression(max_iter=300, solver='saga')
logreg.fit(X_train, y_train)
end = time.time()

# 9. Previsões
y_pred = logreg.predict(X_test)

# 10. Avaliação (holdout)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== Logistic Regression + PCA via SVD ===")
print(f"Accuracy (holdout): {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Tempo de treino: {end - start:.4f} segundos\n")

print("Relatório completo:")
print(classification_report(y_test, y_pred, zero_division=0))

# 11. Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Sem', 'Com'], yticklabels=['Sem', 'Com'])
plt.title("Matriz de Confusão – Logistic Regression com PCA via SVD")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# 12. Validação Cruzada
print("=== Cross-Validation (5-fold) ===")
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_scores = {}
for score in scoring:
    scores = cross_val_score(logreg, X_pca, y, cv=5, scoring=score)
    cv_scores[score] = scores
    print(f"{score}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# 13. Gráfico de comparação das métricas de CV
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
means = [np.mean(cv_scores[s]) for s in scoring]
stds = [np.std(cv_scores[s]) for s in scoring]

plt.figure(figsize=(8, 5))
sns.barplot(x=metrics, y=means, palette="Purples_d", edgecolor="black", capsize=0.2)
plt.errorbar(x=metrics, y=means, yerr=stds, fmt='none', c='black', capsize=5)
plt.title("Cross-Validation (5-fold): Logistic Regression com PCA via SVD")
plt.ylabel("Mean Metric Value")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# NOVO: SALVAR O MODELO E OS PRÉ-PROCESSADORES NECESSÁRIOS
try:
    joblib.dump(logreg, "templates/modelo_classificacao_fertilizante.pkl")
    joblib.dump(scaler_classificacao, "templates/scaler_classificacao.pkl")
    joblib.dump(label_encoders_classificacao, "templates/label_encoders_classificacao.pkl")
    joblib.dump(pca_components, "templates/pca_components.pkl")
    joblib.dump(pca_mean, "templates/pca_mean.pkl")
    joblib.dump(feature_cols_classification, "templates/feature_cols_classification.pkl")
    print("\nModelo de classificação de fertilizante e pré-processadores salvos com sucesso.")
except Exception as e:
    print(f"\nErro ao salvar os arquivos .pkl: {e}")