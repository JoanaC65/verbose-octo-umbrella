import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from math import log
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análise de Dados Agrícolas", page_icon="🌾", layout="wide")

# --- Dados ---
@st.cache_data
def carregar_dados():
    df = pd.read_csv("crop_yield.csv").head(20000)
    return df

df = carregar_dados()

# Novas variáveis
df['Razao_Chuva_Temperatura'] = df['Precipitacao_mm'] / (df['Temperatura_Celsius'] + 1)
df['Rendimento_Por_Dia'] = df['Rendimento_Toneladas_Por_Hectare'] / df['Dias_para_Colheita']

# Codificar categorias
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Remover outliers
df_num = df.select_dtypes(include=[np.number])
Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 - Q1
mask = ~((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).any(axis=1)
df_clean = df[mask].copy()

df_clean["Regiao_Nome"] = label_encoders["Regiao"].inverse_transform(df_clean["Regiao"])
df_clean["Cultura"] = label_encoders["Cultura"].inverse_transform(df_clean["Cultura"])

# Função índice de Shannon
def shannon_index(counts):
    total = sum(counts)
    proportions = [c / total for c in counts if c > 0]
    return -sum(p * log(p) for p in proportions)

diversity = df_clean.groupby("Regiao").apply(lambda g: shannon_index(g["Cultura"].value_counts().values)).reset_index(name="Shannon_Index")
diversity["Regiao_Nome"] = label_encoders["Regiao"].inverse_transform(diversity["Regiao"])

# Tema global para gráficos plotly
plotly_template = "plotly_dark"

# --- Layout Dashboard com abas ---
st.title("🌾 Dashboard: Análise de Dados Agrícolas")

tabs = st.tabs([
    "Matriz de Correlação",
    "Outliers",
    "Distribuição Culturas/Regiões",
    "Rendimento por Cultura",
    "Diversidade Agrícola",
    "Clima (Temp & Chuva)",
    "Relações Temperatura x Rendimento",
    "Gráfico de Bolhas",
    "Medidas Estatísticas"
])

with tabs[0]:
    st.subheader("Matriz de Correlação")
    corr = df.corr()
    fig = px.imshow(corr, color_continuous_scale='RdBu_r', color_continuous_midpoint=0,
                    title="Matriz de Correlação", labels=dict(x="Variáveis", y="Variáveis", color="Correlação"), aspect="auto")
    fig.update_layout(width=900, height=900, template=plotly_template)
    fig.update_traces(text=corr.round(2), texttemplate="%{text}", hoverinfo="text+z")
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Detecção de Outliers (IQR)")
    st.write(f"Tamanho original: {df.shape} | Sem outliers: {df_clean.shape}")
    fig = px.box(df_clean, x="Cultura", y="Rendimento_Toneladas_Por_Hectare", color="Cultura",
                 title="Boxplot Rendimento por Cultura", template=plotly_template)
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Distribuição das Culturas por Região")
    fig = px.histogram(df_clean, x="Regiao_Nome", color="Cultura", barmode="group",
                       title="Culturas por Região", template=plotly_template)
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Rendimento por Cultura")
    fig = px.violin(df_clean, x="Cultura", y="Rendimento_Toneladas_Por_Hectare", box=True, points="all", color="Cultura",
                    title="Rendimento por Cultura", template=plotly_template)
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Diversidade Agrícola por Região (Índice de Shannon)")
    fig = px.bar(diversity, x="Regiao_Nome", y="Shannon_Index", color="Shannon_Index", color_continuous_scale="magma",
                 title="Índice de Diversidade", template=plotly_template)
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("Distribuição de Temperatura e Precipitação")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_clean, x="Temperatura_Celsius", nbins=30, color_discrete_sequence=["#1f77b4"],
                           title="Temperatura (°C)", template=plotly_template)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df_clean, x="Precipitacao_mm", nbins=30, color_discrete_sequence=["#2ca02c"],
                           title="Precipitação (mm)", template=plotly_template)
        st.plotly_chart(fig, use_container_width=True)

with tabs[6]:
    st.subheader("Relação Temperatura x Rendimento por Cultura")
    culturas_selecionadas = st.multiselect("Selecione culturas", options=df_clean["Cultura"].unique(),
                                          default=df_clean["Cultura"].unique()[:1])
    if culturas_selecionadas:
        df_filtrado = df_clean[df_clean["Cultura"].isin(culturas_selecionadas)]
        fig = px.scatter(df_filtrado, x="Temperatura_Celsius", y="Rendimento_Toneladas_Por_Hectare",
                         facet_row="Cultura", color="Cultura", opacity=0.7,
                         title="Temperatura vs. Rendimento por Cultura", template=plotly_template,
                         labels={"Temperatura_Celsius": "Temperatura (°C)", "Rendimento_Toneladas_Por_Hectare": "Rendimento (Toneladas/Ha)"},
                         height=300 * len(culturas_selecionadas))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecione pelo menos uma cultura para exibir o gráfico.")

with tabs[7]:
    st.subheader("Gráfico de Bolhas: Temperatura vs. Precipitação por Cultura")
    cultura_selecionada = st.selectbox("Selecione uma cultura", sorted(df_clean["Cultura"].unique()))
    df_bolha = df_clean[df_clean["Cultura"] == cultura_selecionada]
    df_bolha["Rendimento_Toneladas_Por_Hectare"] = df_bolha["Rendimento_Toneladas_Por_Hectare"].clip(lower=0)
    fig = px.scatter(df_bolha, x="Temperatura_Celsius", y="Precipitacao_mm",
                     size="Rendimento_Toneladas_Por_Hectare", color="Regiao_Nome", facet_row="Regiao_Nome",
                     title=f"Temperatura vs. Precipitação - {cultura_selecionada}", size_max=60,
                     labels={"Temperatura_Celsius": "Temperatura (°C)", "Precipitacao_mm": "Precipitação (mm)", "Rendimento_Toneladas_Por_Hectare": "Rendimento (Toneladas/Ha)"},
                     template=plotly_template)
    fig.update_layout(height=600 * len(df_bolha["Regiao_Nome"].unique()), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tabs[8]:
    st.subheader("Medidas Estatísticas das Variáveis Numéricas")
    variaveis = df_clean.select_dtypes(include=[np.number]).drop(columns=['Cluster'], errors='ignore')
    skewness = variaveis.skew()
    kurt = variaveis.kurtosis()
    coef_var = variaveis.std() / variaveis.mean()
    moda = variaveis.mode().iloc[0]
    erro_padrao = variaveis.sem()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(skewness.reset_index(), x="index", y=0, title="Skewness (Assimetria)", color_discrete_sequence=["#1f77b4"], template=plotly_template)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(kurt.reset_index(), x="index", y=0, title="Curtose", color_discrete_sequence=["#ff7f0e"], template=plotly_template)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.dataframe(pd.DataFrame({
            "Coeficiente de Variação": coef_var.round(3),
            "Erro Padrão": erro_padrao.round(3),
            "Moda": moda.round(3)
        }).style.background_gradient(cmap="viridis"))
    with col4:
        st.write("**Descrição rápida:**\n- Skewness indica simetria da distribuição\n- Curtose indica achatamento\n- Coef. de Variação indica variabilidade relativa")

