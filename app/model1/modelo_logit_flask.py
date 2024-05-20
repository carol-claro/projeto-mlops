import pandas as pd
import zipfile
import io
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)

# Carregar os dados
url = 'https://storage.googleapis.com/ds-publico/IA/loan_default.csv.zip'
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
    with zip_file.open('loan_default.csv') as csv_file:
        df = pd.read_csv(csv_file)

# Selecionar colunas de interesse
keep = ['Status',
        'approv_in_adv',
        'credit_type',
        'loan_purpose',
        'age',
        'co_applicant_credit_type',
        'submission_of_application',
        'lump_sum_payment',
        'loan_amount',
        'income',
        'Credit_Score']

df_modelo = df[keep]

# Fazer limpeza nos dados
df_modelo = df_modelo.query("income != 0").dropna(subset=['income'])
df_modelo = df_modelo.dropna(subset=['loan_purpose'])
df_modelo = df_modelo.dropna(subset=['approv_in_adv'])

# Separar variáveis independentes e dependentes
x = df_modelo.drop(columns=['Status'])
y = df_modelo['Status']

# Transformar variáveis contínuas em logaritmo natural
x['loan_amount'] = np.log(x['loan_amount'])
x['income'] = np.log(x['income'])

# Transformar variáveis categóricas em dummies
categoricas = ['approv_in_adv',
            'credit_type',
            'loan_purpose',
            'age',
            'co_applicant_credit_type',
            'submission_of_application',
            'lump_sum_payment']

x = pd.get_dummies(x, columns=categoricas, drop_first=True)

# Separar conjunto de dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão logística
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(x_train, y_train)

# Calcular a probabilidade ótima para o corte
probs_train = logistic_model.predict_proba(x_train)
fpr, tpr, thresholds = roc_curve(y_train, probs_train[:, 1])
distancia = np.sqrt(fpr**2 + (1 - tpr)**2)
limiar_otimo = thresholds[np.argmin(distancia)]

print("Limiar ótimo de probabilidade:", limiar_otimo)

# Salvar o modelo treinado e o limiar ótimo em um arquivo usando pickle
with open('logistic_model.pkl', 'wb') as model_file:
    pickle.dump((logistic_model, limiar_otimo, x_train.columns.tolist()), model_file)

# Rota para prever a probabilidade de inadimplência
@app.route('/predict_logistic_regression', methods=['POST'])
def predict_logistic_regression():
    # Carregar o modelo treinado e o limiar ótimo
    with open('logistic_model.pkl', 'rb') as model_file:
        model, limiar_otimo, columns = pickle.load(model_file)

    # Receber dados da requisição
    data = request.json

    # Converter os dados em DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Aplicar as mesmas transformações
    input_data['loan_amount'] = np.log(input_data['loan_amount'])
    input_data['income'] = np.log(input_data['income'])
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Garantir que todas as colunas de dummy estejam presentes
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reordenar as colunas para corresponder ao modelo treinado
    input_data = input_data[columns]

    # Fazer a previsão das probabilidades
    prediction_probs = model.predict_proba(input_data)

    # Aplicar o limiar ótimo para classificar
    prediction_class = (prediction_probs[:, 1] >= limiar_otimo).astype(int)

    # Preparar a resposta
    response = {
        'probabilities': prediction_probs.tolist(),
        'class': prediction_class.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
