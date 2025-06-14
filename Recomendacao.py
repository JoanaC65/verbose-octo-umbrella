import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Carregar os dados
df = pd.read_csv("crop_yield.csv")

# Copiar para não alterar o original
data = df.copy()

# Codificar variáveis categóricas
cat_cols = ['Regiao', 'Tipo_Solo', 'Cultura', 'Condicao_Climatica']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Codificar booleanos
data['Fertilizante'] = data['Fertilizante'].astype(int)
data['Irrigacao'] = data['Irrigacao'].astype(int)

# Normalizar os dados
features = ['Regiao', 'Tipo_Solo', 'Cultura', 'Precipitacao_mm', 'Temperatura_Celsius',
            'Condicao_Climatica', 'Dias_para_Colheita']
scaler = StandardScaler()
X = scaler.fit_transform(data[features])

# ➕ Simular entrada do usuário
nova_entrada = {
    'Regiao': 'East',
    'Tipo_Solo': 'Sandy',
    'Cultura': 'Wheat',
    'Precipitacao_mm': 850.0,
    'Temperatura_Celsius': 27.0,
    'Condicao_Climatica': 'Cloudy',
    'Dias_para_Colheita': 115
}

entrada_codificada = [
    encoders['Regiao'].transform([nova_entrada['Regiao']])[0],
    encoders['Tipo_Solo'].transform([nova_entrada['Tipo_Solo']])[0],
    encoders['Cultura'].transform([nova_entrada['Cultura']])[0],
    nova_entrada['Precipitacao_mm'],
    nova_entrada['Temperatura_Celsius'],
    encoders['Condicao_Climatica'].transform([nova_entrada['Condicao_Climatica']])[0],
    nova_entrada['Dias_para_Colheita']
]

entrada_normalizada = scaler.transform([entrada_codificada])

# Calcular similaridade
similaridades = cosine_similarity(X, entrada_normalizada).flatten()

# Obter os top 5 mais similares
top_k_idx = similaridades.argsort()[-5:][::-1]
top_k = data.iloc[top_k_idx]

# Mostrar as melhores práticas entre os mais similares
print("Práticas recomendadas com base em registros semelhantes:")
print(top_k[['Fertilizante', 'Irrigacao', 'Rendimento_Toneladas_Por_Hectare']])

# Recomendar prática mais comum entre os mais produtivos
melhores = top_k.sort_values(by='Rendimento_Toneladas_Por_Hectare', ascending=False).head(3)
fertilizante_recomendado = melhores['Fertilizante'].mode()[0]
irrigacao_recomendada = melhores['Irrigacao'].mode()[0]

print("\nRecomendação final:")
print(f"Usar fertilizante? {'Sim' if fertilizante_recomendado else 'Não'}")
print(f"Usar irrigação? {'Sim' if irrigacao_recomendada else 'Não'}")
