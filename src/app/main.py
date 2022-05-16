from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle
import os

# Nesse arquivo, trago diretamente o modelo de regressão linear. Não vou fazer o treinamento dele,
# dentro da API.

# request: usa para o método POST
# jsonify: usa para trabalhar com json na saída do modelo

# Carregando o modelo
modelo = pickle.load(open('../../models/modelo.sav', 'rb'))

# Criando o App
app = Flask(__name__)

# Configuração de Autenticação
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')
basicauth = BasicAuth(app)

# Passando as colunas que usamos no modelo, pra depois usar no json
colunas = ['tamanho', 'ano', 'garagem']

# Home: é definida com '/'
@app.route('/')
def home():
    return('API para previsao de preços de casas!')

# Página análise de sentimento
@app.route('/sentimento/<frase>')
@basicauth.required
def sentimento(frase):
    tb = TextBlob(frase)
    polaridade = tb.sentiment.polarity
    return f"polaridade: {polaridade}"

# Página cotacao da casa
@app.route('/cotacao/', methods = ['POST'])
@basicauth.required
def cotacao():
    # Essa linha abaixo, recebe o valor json
    dados = request.get_json()

    # Essa linha abaixo, garante que a ordem do json vai estar certa.
    dados_input = [dados[col] for col in colunas]
    
    # Aqui fazemos a previsão do modelo.
    preco = modelo.predict([dados_input])
    
    return jsonify(preco = round(preco[0], 2))

app.run(debug = True, host = '0.0.0.0')