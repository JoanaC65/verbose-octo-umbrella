import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from DashboardML import cols_treinadas

# 1. Carregar os dados
df = pd.read_csv("crop_yield.csv").head(100000)

# 2. Codificar variáveis categóricas com OrdinalEncoder
cat_cols = df.select_dtypes(include='object').columns.tolist()
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[cat_cols] = encoder.fit_transform(df[cat_cols])

# 3. Separar variáveis
X = df.drop(columns=['Rendimento_Toneladas_Por_Hectare'])
y = df['Rendimento_Toneladas_Por_Hectare']

# 4. Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Colunas esperadas:", cols_treinadas)
print("Colunas atuais do df:", df.columns.tolist())


# 5. Pipeline: StandardScaler + PCA + Ridge
pca_ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, svd_solver='full')),
    ('ridge', Ridge(alpha=1.0))
])

# 6. Treinamento
start = time.time()
pca_ridge_pipeline.fit(X_train, y_train)
end = time.time()

print("Categorias mapeadas por coluna:")
for i, col in enumerate(cat_cols):
    print(f"{col}: {encoder.categories_[i]}")

# 7. Previsão
y_pred = pca_ridge_pipeline.predict(X_test)
residuals = y_test - y_pred

# 8. Avaliação (Holdout)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("=== Ridge + PCA (SVD) ===")
print(f"R² (holdout): {r2:.4f}")
print(f"RMSE (holdout): {rmse:.4f}")
print(f"MAE (holdout): {mae:.4f}")
print(f"Tempo de treino: {end - start:.2f} segundos")

# 9. Validação cruzada
print("\n=== Validação Cruzada (5-fold) ===")
scores_r2 = cross_val_score(pca_ridge_pipeline, X, y, cv=5, scoring='r2')
scores_rmse = cross_val_score(pca_ridge_pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
scores_mae = cross_val_score(pca_ridge_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

print(f"R² médio: {np.mean(scores_r2):.4f}")
print(f"RMSE médio: {-np.mean(scores_rmse):.4f}")
print(f"MAE médio: {-np.mean(scores_mae):.4f}")

# 10. Visualizações
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal (y = x)')
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("PCA + Ridge: Previsão vs Real")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30, color='steelblue')
plt.title("Distribuição dos Erros (Resíduos)")
plt.xlabel("Erro (Real - Previsto)")
plt.tight_layout()
plt.show()

# 11. Salvar arquivos
joblib.dump(pca_ridge_pipeline, "templates/modelo_pca_ridge.pkl")
joblib.dump(encoder, "templates/ordinal_encoder.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(y_pred, "y_pred.pkl")
joblib.dump(X.columns.tolist(), "templates/colunas_treinadas.pkl")

print("Modelos e ficheiros salvos com sucesso.")
