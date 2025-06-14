import io
import joblib
import numpy as np
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from scipy.stats import kurtosis, skew
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
import openai
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)

# Chave da API OpenAI
openai.api_key = ""


# Carregar dataset
df = pd.read_csv("crop_yield.csv").head(100000)
print(df.head())

# Criar variáveis derivadas
df['Razao_Chuva_Temperatura'] = df['Precipitacao_mm'] / (df['Temperatura_Celsius'] + 1)
df['Rendimento_Por_Dia'] = df['Rendimento_Toneladas_Por_Hectare'] / df['Dias_para_Colheita']

# Codificar variáveis categóricas
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Variáveis categóricas (ajuste conforme necessário)
cat_vars = ["Irrigacao", "Fertilizante", "Regiao", "Tipo_Solo", "Cultura", "Condicao_Climatica"]

# Lista de todas as variáveis disponíveis
variaveis = df.columns.tolist()

# --- CARREGAMENTO DE MODELOS E DADOS PARA A PREVISÃO (DO SEGUNDO ARQUIVO) ---
# df_predict será usado para a lógica de previsão e recomendação
df_predict = pd.read_csv("crop_yield.csv")
data_predict = df_predict.copy() # Cópia para encoding


# **ADICIONE ESTA VERIFICAÇÃO AQUI**
if 'Rendimento_Toneladas_Por_Hectare' not in data_predict.columns:
    print("ERRO CRÍTICO: 'Rendimento_Toneladas_Por_Hectare' não encontrada em data_predict!")
    # Você pode até forçar o app a parar ou retornar um erro se quiser ser mais rigoroso
    # Ou tentar carregar uma versão "segura" do CSV
    # exit("Coluna de Rendimento ausente, impossível continuar.")
else:
    print("DEBUG: 'Rendimento_Toneladas_Por_Hectare' encontrada em data_predict.")

# Carregar modelo e encoders salvos
model = joblib.load("templates/modelo_pca_ridge.pkl")

# Carregar modelo e encoders salvos
#model = joblib.load("templates/modelo_pca_ridge.pkl")
# O 'encoder' aqui é para as colunas do modelo (OrdinalEncoder se for o caso, mas seu exemplo usa LabelEncoder)
# Se você salvou um OrdinalEncoder, certifique-se de que o nome do arquivo corresponde
encoder_model = joblib.load("templates/ordinal_encoder.pkl") # Se for OrdinalEncoder
cols_treinadas = joblib.load("templates/colunas_treinadas.pkl")

# Encoders para colunas categóricas para a previsão e recomendação
cat_cols_predict = ['Regiao', 'Tipo_Solo', 'Cultura', 'Condicao_Climatica']
encoders_predict = {}
for col in cat_cols_predict:
    le = LabelEncoder()
    data_predict[col] = le.fit_transform(data_predict[col])
    encoders_predict[col] = le

# O solo_encoder é o LabelEncoder específico para 'Tipo_Solo' do df_predict
solo_encoder = encoders_predict['Tipo_Solo']

data_predict['Fertilizante'] = data_predict['Fertilizante'].astype(int)
data_predict['Irrigacao'] = data_predict['Irrigacao'].astype(int)

features_predict = ['Regiao', 'Tipo_Solo', 'Cultura', 'Precipitacao_mm', 'Temperatura_Celsius',
                    'Condicao_Climatica', 'Dias_para_Colheita']
scaler_predict = StandardScaler()
X_predict = scaler_predict.fit_transform(data_predict[features_predict])


try:
    model_classificacao_fertilizante = joblib.load("templates/modelo_classificacao_fertilizante.pkl")
    scaler_classificacao_fertilizante = joblib.load("templates/scaler_classificacao.pkl")
    label_encoders_classificacao_fertilizante = joblib.load("templates/label_encoders_classificacao.pkl")
    pca_components_fertilizante = joblib.load("templates/pca_components.pkl")
    pca_mean_fertilizante = joblib.load("templates/pca_mean.pkl")
    feature_cols_classificacao_fertilizante = joblib.load("templates/feature_cols_classification.pkl")
    print("Modelo de classificação de fertilizante e pré-processadores carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar modelo de classificação de fertilizante ou seus pré-processadores: {e}")
    model_classificacao_fertilizante = None
    scaler_classificacao_fertilizante = None
    label_encoders_classificacao_fertilizante = {}
    pca_components_fertilizante = None
    pca_mean_fertilizante = None
    feature_cols_classificacao_fertilizante = []


# --- FUNÇÕES AUXILIARES (UNIFICADO) ---
# Função GPT
def obter_resposta_gpt(pergunta):
    prompt = f"Eu tenho os seguintes dados de um arquivo CSV:\n{df.head()}\n\nPergunta: {pergunta}\nResposta:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um assistente que responde perguntas sobre dados CSV."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']
# Função de recomendação (do segundo arquivo)
def recomendar_praticas(entrada):
    # Usar encoders_predict e scaler_predict para a lógica de previsão
    entrada_codificada = [
        encoders_predict['Regiao'].transform([entrada['Regiao']])[0],
        encoders_predict['Tipo_Solo'].transform([entrada['Tipo_Solo']])[0],
        encoders_predict['Cultura'].transform([entrada['Cultura']])[0],
        entrada['Precipitacao_mm'],
        entrada['Temperatura_Celsius'],
        encoders_predict['Condicao_Climatica'].transform([entrada['Condicao_Climatica']])[0],
        entrada['Dias_para_Colheita']
    ]
    entrada_normalizada = scaler_predict.transform([entrada_codificada])

    similaridades = cosine_similarity(X_predict, entrada_normalizada).flatten()
    top_k_idx = similaridades.argsort()[-10:][::-1]
    top_k = data_predict.iloc[top_k_idx]

    melhores_rendimento = top_k.sort_values(by='Rendimento_Toneladas_Por_Hectare', ascending=False).head(5)

    fertilizante_recomendado = bool(melhores_rendimento['Fertilizante'].mode()[0])
    irrigacao_recomendada = bool(melhores_rendimento['Irrigacao'].mode()[0])

    cultura_especifica = entrada['Cultura']
    solo_recomendado_para_cultura = None

    df_cultura = df_predict[df_predict['Cultura'] == cultura_especifica] # Usa df_predict aqui

    if not df_cultura.empty:
        top_solos_cultura = df_cultura.sort_values(by='Rendimento_Toneladas_Por_Hectare', ascending=False).head(3)
        if not top_solos_cultura.empty:
            # Pega o nome do solo direto do df_predict (que tem strings)
            solo_recomendado_para_cultura = top_solos_cultura['Tipo_Solo'].mode()[0]

    return {
        'usar_fertilizante': fertilizante_recomendado,
        'usar_irrigacao': irrigacao_recomendada,
        'melhor_tipo_solo': solo_recomendado_para_cultura
    }

# ✅ Endpoint compartilhado pelas duas páginas HTML
@app.route('/pergunta', methods=['POST'])
def responder_pergunta():
    pergunta = request.json.get('pergunta', '')
    if not pergunta:
        return jsonify({'resposta': 'Por favor, faça uma pergunta.'})
    try:
        resposta = obter_resposta_gpt(pergunta)
        return jsonify({'resposta': resposta})
    except Exception as e:
        return jsonify({'resposta': f"Erro ao processar a pergunta: {str(e)}"})

# Rota para a capa inicial
@app.route('/')
def capa():
    return render_template('CapaInicial.html')

# Rota para a página de previsão (renderiza o HTML)
# Rota para a página de previsão (renderiza o HTML)
@app.route('/predict_page') # Renomeei para evitar conflito com o POST /predict
def previsao_page():
    return render_template('previsaoOFICIAL.html')

# Endpoint para a previsão (POST request do formulário de previsão)
@app.route('/predict', methods=['POST'])
def predict():
    # ... sua lógica de previsão e recomendação ...
    data = request.json
    print("Dados recebidos para previsão:", data)
    df_input = pd.DataFrame([[
        data['Regiao'], data['Tipo_Solo'], data['Cultura'],
        data['Precipitacao_mm'], data['Temperatura_Celsius'],
        int(data['Fertilizante']), int(data['Irrigacao']),
        data['Condicao_Climatica'], data['Dias_para_Colheita']
    ]], columns=[
        'Regiao', 'Tipo_Solo', 'Cultura', 'Precipitacao_mm', 'Temperatura_Celsius',
        'Fertilizante', 'Irrigacao', 'Condicao_Climatica', 'Dias_para_Colheita'
    ])

    # Encoding categorias (usar o encoder_model para o modelo, que pode ser diferente do `encoders_predict` que é para a similaridade)
    # Verifique qual encoder é o correto para o seu modelo carregado
    encoded_cols = encoder_model.transform(df_input[['Regiao', 'Tipo_Solo', 'Cultura', 'Condicao_Climatica']])
    df_input[['Regiao', 'Tipo_Solo', 'Cultura', 'Condicao_Climatica']] = encoded_cols

    # Organizar colunas (garantir ordem para o modelo)
    df_input = df_input[cols_treinadas]

    pred = model.predict(df_input)[0]
    print("Previsão calculada:", pred)

    # Gerar recomendação com dados originais (antes da codificação)
    recomendacao = recomendar_praticas(data) # 'data' é o input original do formulário

    # Salvar para gerar relatório depois
    app.config['ULTIMO_RESULTADO'] = {
        'input_data': data,
        'previsao': pred,
        'recomendacao': recomendacao
    }

    # Retornar previsão e recomendação juntas no JSON
    return jsonify({
        'previsao': round(pred, 2),
        'recomendacao': recomendacao
    })

#Classificação
@app.route('/predict_fertilizante', methods=['POST'])
def predict_fertilizante():
    data = request.json
    print("Dados recebidos para previsão de fertilizante:", data)

    if not (model_classificacao_fertilizante and scaler_classificacao_fertilizante and
            pca_components_fertilizante is not None and pca_mean_fertilizante is not None and
            feature_cols_classificacao_fertilizante):
        return jsonify({"erro": "Modelo de classificação de fertilizante não carregado ou incompleto."}), 500

    try:
        # Criar DataFrame com os dados de entrada para o modelo de classificação
        # As colunas de entrada para o modelo de classificação são:
        # 'Regiao', 'Tipo_Solo', 'Cultura', 'Precipitacao_mm', 'Temperatura_Celsius',
        # 'Irrigacao', 'Condicao_Climatica', 'Dias_para_Colheita'
        # Note que 'Fertilizante' NÃO deve estar aqui, pois é a variável alvo.

        input_for_fertilizer_prediction = {
            'Regiao': data.get('Regiao'),
            'Tipo_Solo': data.get('Tipo_Solo'),
            'Cultura': data.get('Cultura'),
            'Precipitacao_mm': float(data.get('Precipitacao_mm')),
            'Temperatura_Celsius': float(data.get('Temperatura_Celsius')),
            'Irrigacao': int(data.get('Irrigacao')), # Assumindo que vem como booleano ou string '0'/'1'
            'Condicao_Climatica': data.get('Condicao_Climatica'),
            'Dias_para_Colheita': int(data.get('Dias_para_Colheita')),
            'Rendimento_Toneladas_Por_Hectare': float(data.get('Rendimento_Toneladas_Por_Hectare', 0.0))
        }

        # Criar um DataFrame a partir dos dados de entrada
        df_input_classificacao = pd.DataFrame([input_for_fertilizer_prediction])

        # Codificar variáveis categóricas para o modelo de classificação
        for col, le in label_encoders_classificacao_fertilizante.items():
            if col in df_input_classificacao.columns:
                # Verifique se o valor de entrada existe nas classes do encoder
                if df_input_classificacao[col].iloc[0] not in le.classes_:
                    return jsonify({"erro": f"Valor '{df_input_classificacao[col].iloc[0]}' para '{col}' não reconhecido pelo modelo de fertilizante."}), 400
                df_input_classificacao[col] = le.transform(df_input_classificacao[col])

        # Garantir a ordem das colunas para o modelo de classificação
        df_input_classificacao = df_input_classificacao[feature_cols_classificacao_fertilizante]

        # Padronizar os dados
        input_scaled = scaler_classificacao_fertilizante.transform(df_input_classificacao)

        # Aplicar PCA manualmente (usando os componentes salvos)
        input_centered_pca = input_scaled - pca_mean_fertilizante
        input_pca = np.dot(input_centered_pca, pca_components_fertilizante.T)

        # Fazer a previsão
        pred_fertilizante = model_classificacao_fertilizante.predict(input_pca)[0]
        prob_fertilizante = model_classificacao_fertilizante.predict_proba(input_pca)[0].tolist() # Probabilidades

        resultado_fertilizante = "Sim" if pred_fertilizante == 1 else "Não"

        return jsonify({
            'fertilizante_predicao': resultado_fertilizante,
            'probabilidade_nao': round(prob_fertilizante[0], 4),
            'probabilidade_sim': round(prob_fertilizante[1], 4)
        })

    except ValueError as ve:
        return jsonify({"erro": f"Erro de valor ou tipo de dado: {str(ve)}. Verifique se todos os campos estão preenchidos corretamente e com o tipo esperado (ex: números para Precipitação)."}), 400
    except Exception as e:
        print(f"Erro inesperado na previsão de fertilizante: {e}")
        return jsonify({"erro": f"Erro interno ao processar a previsão de fertilizante: {str(e)}"}), 500


# Página principal com estatísticas
@app.route('/index2')
def index2():
    return render_template('index2.html', variaveis=variaveis)

# Página alternativa com API (por exemplo, chatbot)
@app.route('/pagina2API')
def pagina2API():
    return render_template('pagina2API.html', variaveis=variaveis)

# Rota para a página de previsão de fertilizante (renderiza o HTML)
@app.route('/fertilizante_page')
def fertilizante_page():
    return render_template('classificacao2.html')

# Estatísticas completas (para /estatisticas)
@app.route('/estatisticas', methods=['GET', 'POST'])
def estatisticas():
    variavel = request.form.get('variavel')
    if not variavel or variavel not in df.columns:
        return render_template('index2.html', variaveis=variaveis, erro="Variável não encontrada")

    dtype_info = pd.DataFrame({"Variável": [variavel], "Tipo": [str(df[variavel].dtype)]}).to_html(index=False)

    if variavel in cat_vars:
        groups = df.groupby(variavel).size().reset_index(name='Contagem')
        groups['Percentual'] = (groups['Contagem'] / len(df)) * 100
        if variavel in label_encoders:
            le = label_encoders[variavel]
            groups['Categoria'] = groups[variavel].apply(lambda x: le.inverse_transform([x])[0])
        group_stats = "<h3>Estatísticas de Grupo (Contagem, Percentual e Categoria)</h3>" + groups.to_html(index=False)
    else:
        group_stats = "<p>Não é uma variável categórica.</p>"

    if np.issubdtype(df[variavel].dtype, np.number) and variavel not in cat_vars and df[variavel].dtype != bool:
        basic_stats = df[variavel].describe().to_frame().T.to_html(index=False)
        col_numeric = df[variavel]
        media = col_numeric.mean()
        coef_var = col_numeric.std() / media if media != 0 else np.nan
        numeric_stats_df = pd.DataFrame({
            "Skewness": [skew(col_numeric)],
            "Kurtosis": [kurtosis(col_numeric)],
            "Coeficiente de Variação": [coef_var],
            "Moda": [col_numeric.mode().iloc[0] if not col_numeric.mode().empty else np.nan],
            "Erro Padrão da Média": [col_numeric.sem()]
        })
        html_table = numeric_stats_df.to_html(index=False)

        tooltips = {
            "Skewness": "Distribuição assimétrica. Indica a frequência de valores menores e maiores.(positiva-valores menores; negativa-valores maiores).",
            "Kurtosis": "Quantifica outliers. Se >3 existem mais outliers. Se <3 existem menos outliers .",
            "Coeficiente de Variação": "Dispersão relativa. Quanto mai elevado maior a variabilidade em torno da média",
            "Moda": "Valor mais frequente.",
            "Erro Padrão da Média": "Precisão da média. Descreve a incerteza da amostra"
        }
        for key, tooltip in tooltips.items():
            html_table = html_table.replace(
                key,
                f'<span data-bs-toggle="tooltip" title="{tooltip}">{key} ℹ️</span>'
            )
        numeric_stats = html_table
    else:
        moda = df[variavel].mode().iloc[0] if not df[variavel].mode().empty else np.nan
        numeric_stats = f'<p>Moda: {moda}</p>'
        basic_stats = "<p>Não disponível para variáveis categóricas e booleanas.</p>"

    return render_template('index2.html', variaveis=variaveis, dtype=dtype_info, group_stats=group_stats,
                           basic_stats=basic_stats, numeric_stats=numeric_stats)

# Função auxiliar para estatísticas normalizadas/padronizadas
def calcular_estatisticas_variavel(transform_func, variavel):
    if variavel not in df.columns:
        return {"erro": "Variável não encontrada"}
    if variavel in cat_vars or df[variavel].dtype == bool:
        return {"erro": "Estatísticas não disponíveis para variáveis categóricas ou booleanas"}

    data = df[variavel].astype(float)
    transformed = transform_func(data)
    series = pd.Series(transformed)

    stats_basicas = {
        "Variável": variavel,
        "Média": series.mean(),
        "Desvio Padrão": series.std(),
        "Mínimo": series.min(),
        "25%": series.quantile(0.25),
        "50% (Mediana)": series.median(),
        "75%": series.quantile(0.75),
        "Máximo": series.max(),
    }

    stats_avancadas = {
        "Skewness": float(skew(series)),
        "Kurtosis": float(kurtosis(series)),
        "Coeficiente de Variação": series.std() / series.mean() if series.mean() != 0 else None,
        "Moda": series.mode().iloc[0] if not series.mode().empty else None,
        "Erro Padrão da Média": series.sem()
    }

    return {"estatisticas_basicas": stats_basicas, "estatisticas_avancadas": stats_avancadas}

# Endpoint para normalização
@app.route('/stats_normalizado_var', methods=['GET'])
def stats_normalizado_var():
    variavel = request.args.get('variavel')
    if not variavel:
        return jsonify({"erro": "Nenhuma variável fornecida"})

    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    stats = calcular_estatisticas_variavel(normalize, variavel)
    return jsonify(stats)

# Endpoint para padronização
@app.route('/stats_padronizado_var', methods=['GET'])
def stats_padronizado_var():
    variavel = request.args.get('variavel')
    if not variavel:
        return jsonify({"erro": "Nenhuma variável fornecida"})

    def standardize(data):
        return (data - np.mean(data)) / np.std(data)

    stats = calcular_estatisticas_variavel(standardize, variavel)
    return jsonify(stats)

# Endpoint para gerar relatório PDF (do segundo arquivo)
@app.route('/gerar_relatorio')
def gerar_relatorio():
    data = app.config.get('ULTIMO_RESULTADO')
    if not data:
        return "Nenhum resultado para exportar.", 400

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Relatório de Previsão de Rendimento Agrícola")

    # Dados de entrada
    c.setFont("Helvetica", 12)
    y = height - 100
    c.drawString(50, y, "📥 Dados de entrada:")
    for key, val in data['input_data'].items():
        y -= 20
        c.drawString(70, y, f"{key}: {val}")

    # Previsão
    y -= 40
    c.drawString(50, y, f"🔍 Previsão: {data['previsao']:.2f} toneladas por hectare")

    # Recomendações dinâmicas
    y -= 40
    c.drawString(50, y, "💡 Recomendações:")
    recs = []
    if data['recomendacao']['usar_fertilizante']:
        recs.append("Usar Fertilizante.")
    else:
        recs.append("O uso de fertilizante pode não ser necessário.")
    if data['recomendacao']['usar_irrigacao']:
        recs.append("Usar Irrigação.")
    else:
        recs.append("O uso de irrigação pode não ser necessário.")

    if data['recomendacao']['melhor_tipo_solo']:
        recs.append(f"Para {data['input_data']['Cultura']}, o solo com melhor rendimento é: {data['recomendacao']['melhor_tipo_solo']}.")
    else:
        recs.append(f"Não há dados suficientes para recomendar o melhor tipo de solo para {data['input_data']['Cultura']} com base no rendimento.")

    for rec in recs:
        y -= 20
        c.drawString(70, y, f"- {rec}")
    # ... (Restante do código do relatório, como o gráfico) ...
    # Gera gráfico diretamente em memória
    plt.figure(figsize=(4, 3))
    plt.bar(['Previsão', 'Meta'], [data['previsao'], 8], color=['#2e7d32', '#ffc107'])
    plt.title('Comparação com Meta')
    plt.ylabel('t/ha')
    img_buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_buffer, format='PNG')
    plt.close()

    # Insere imagem no PDF
    img_buffer.seek(0)
    c.drawImage(ImageReader(img_buffer), 50, y - 220, width=300, height=200) # Ajuste a posição Y conforme necessário para não sobrepor o texto

    c.showPage()
    c.save()

    buffer.seek(0)
    return send_file(buffer,
                     as_attachment=True,
                     download_name="relatorio_previsao.pdf",
                     mimetype='application/pdf')

# Iniciar app
if __name__ == '__main__':
    app.run(debug=True)
