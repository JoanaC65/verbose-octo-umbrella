import openai
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Configuração da API do OpenAI
openai.api_key = ""  # Coloque sua chave de API do OpenAI

# Inicializar o Flask
app = Flask(__name__)

# Carregar o CSV
df = pd.read_csv('crop_yield.csv')  # Substitua pelo caminho correto para o seu CSV


# Função para processar a pergunta utilizando o modelo GPT
def obter_resposta_gpt(pergunta):
    prompt = f"Eu tenho os seguintes dados em um arquivo CSV:\n{df.head()}\n\nPergunta: {pergunta}\nResposta:"

    response = openai.Completion.create(
        engine="text-davinci-003",  # Ou "gpt-4", caso tenha acesso
        prompt=prompt,
        temperature=0.7,
        max_tokens=150
    )

    return response.choices[0].text.strip()


@app.route('/')
def index():
    return render_template('pagina2API.html')

@app.route('/previsao')
def previsao():
    return render_template('previsao.html')

@app.route('/pergunta', methods=['POST'])
def resposta():
    pergunta = request.json.get('pergunta', '')
    if not pergunta:
        return jsonify({'resposta': 'Por favor, faça uma pergunta.'})

    try:
        resposta = obter_resposta_gpt(pergunta)
        return jsonify({'resposta': resposta})
    except Exception as e:
        return jsonify({'resposta': f'Ocorreu um erro ao processar a pergunta: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True)
