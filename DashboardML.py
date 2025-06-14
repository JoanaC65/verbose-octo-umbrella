import io
from flask import Flask, request, jsonify, send_file, render_template
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('previsao.html')


# Carregar modelo e encoder
model = joblib.load("templates/modelo_pca_ridge.pkl")
encoder = joblib.load("templates/ordinal_encoder.pkl")
cols_treinadas = joblib.load("templates/colunas_treinadas.pkl")

# Carregue e prepare seus dados e encoders (fora da função)
df = pd.read_csv("crop_yield.csv")
data = df.copy()

cat_cols = ['Regiao', 'Tipo_Solo', 'Cultura', 'Condicao_Climatica']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Salvar o LabelEncoder para Tipo_Solo para poder reverter depois
# Isso é importante para mostrar o nome do solo no frontend, e não o número codificado
solo_encoder = encoders['Tipo_Solo']

data['Fertilizante'] = data['Fertilizante'].astype(int)
data['Irrigacao'] = data['Irrigacao'].astype(int)

features = ['Regiao', 'Tipo_Solo', 'Cultura', 'Precipitacao_mm', 'Temperatura_Celsius',
            'Condicao_Climatica', 'Dias_para_Colheita']
scaler = StandardScaler()
X = scaler.fit_transform(data[features])

def recomendar_praticas(entrada):
    entrada_codificada = [
        encoders['Regiao'].transform([entrada['Regiao']])[0],
        encoders['Tipo_Solo'].transform([entrada['Tipo_Solo']])[0],
        encoders['Cultura'].transform([entrada['Cultura']])[0],
        entrada['Precipitacao_mm'],
        entrada['Temperatura_Celsius'],
        encoders['Condicao_Climatica'].transform([entrada['Condicao_Climatica']])[0],
        entrada['Dias_para_Colheita']
    ]
    entrada_normalizada = scaler.transform([entrada_codificada])

    similaridades = cosine_similarity(X, entrada_normalizada).flatten()
    top_k_idx = similaridades.argsort()[-10:][::-1] # Aumentar para 10 para ter mais opções de solo
    top_k = data.iloc[top_k_idx]

    # Filtra os 5 melhores resultados em rendimento (apenas entre os mais similares)
    melhores_rendimento = top_k.sort_values(by='Rendimento_Toneladas_Por_Hectare', ascending=False).head(5)

    fertilizante_recomendado = bool(melhores_rendimento['Fertilizante'].mode()[0])
    irrigacao_recomendada = bool(melhores_rendimento['Irrigacao'].mode()[0])

    # --- NOVA LÓGICA PARA RECOMENDAR TIPO DE SOLO ---
    # Filtra o tipo de solo predominante para a CULTURA específica entre os melhores rendimentos
    # Isso garante que a recomendação seja para a cultura que o usuário selecionou
    cultura_especifica = entrada['Cultura']
    solo_recomendado_para_cultura = None

    # Filtrar o DataFrame original para a cultura e encontrar os tipos de solo com maior rendimento
    df_cultura = df[df['Cultura'] == cultura_especifica]

    if not df_cultura.empty:
        # Encontrar o tipo de solo associado aos 3 maiores rendimentos para essa cultura
        top_solos_cultura = df_cultura.sort_values(by='Rendimento_Toneladas_Por_Hectare', ascending=False).head(3)
        if not top_solos_cultura.empty:
            solo_recomendado_para_cultura = top_solos_cultura['Tipo_Solo'].mode()[0]

    return {
        'usar_fertilizante': fertilizante_recomendado,
        'usar_irrigacao': irrigacao_recomendada,
        'melhor_tipo_solo': solo_recomendado_para_cultura # Adiciona a recomendação de solo
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Dados recebidos:", data)
    df_input = pd.DataFrame([[
        data['Regiao'], data['Tipo_Solo'], data['Cultura'],
        data['Precipitacao_mm'], data['Temperatura_Celsius'],
        int(data['Fertilizante']), int(data['Irrigacao']),
        data['Condicao_Climatica'], data['Dias_para_Colheita']
    ]], columns=[
        'Regiao', 'Tipo_Solo', 'Cultura', 'Precipitacao_mm', 'Temperatura_Celsius',
        'Fertilizante', 'Irrigacao', 'Condicao_Climatica', 'Dias_para_Colheita'
    ])

    # Encoding categorias
    encoded_cols = encoder.transform(df_input[['Regiao', 'Tipo_Solo', 'Cultura', 'Condicao_Climatica']])
    df_input[['Regiao', 'Tipo_Solo', 'Cultura', 'Condicao_Climatica']] = encoded_cols

    # Organizar colunas (garantir ordem)
    df_input = df_input[cols_treinadas]

    pred = model.predict(df_input)[0]
    print("Previsão calculada:", pred)

    # Gerar recomendação com dados originais (antes da codificação)
    recomendacao = recomendar_praticas(data)

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
# ... (código existente) ...

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
    c.drawString(50, y, "Recomendações:")
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

if __name__ == '__main__':
    app.run(debug=True)