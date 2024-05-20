import pandas as pd
import numpy as np
import zipfile
import io
import requests
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

# Variável global para armazenar os dados
df_cluster = None

# Função para carregar os dados e treinar o modelo KMeans
def train_model():
    global df_cluster
    
    url = 'https://storage.googleapis.com/ds-publico/IA/loan_default.csv.zip'
    response = requests.get(url)

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        with zip_file.open('loan_default.csv') as csv_file:
            df = pd.read_csv(csv_file)
            
    # Selecionar as colunas necessárias
    keep = ['Status', 'approv_in_adv', 'credit_type', 'loan_purpose', 'age', 'co_applicant_credit_type',
            'submission_of_application', 'lump_sum_payment', 'loan_amount', 'income', 'Credit_Score']
    df_cluster = df[keep]
    
    # Fazer algumas limpezas nos dados
    df_cluster = df_cluster.query("income != 0").dropna(subset=['income'])
    df_cluster = df_cluster.dropna(subset=['loan_purpose'])
    df_cluster = df_cluster.dropna(subset=['approv_in_adv'])

    # Separar a variável dependente das explicativas
    x = df_cluster.drop(columns=['Status'])
    y = df_cluster['Status']

    # Transformar as variáveis contínuas em logaritmo natural
    x['loan_amount'] = np.log(x['loan_amount'])
    x['income'] = np.log(x['income'])

    # Transformar as variáveis categóricas em dummies
    categoricas = ['approv_in_adv', 'credit_type', 'loan_purpose', 'age', 'co_applicant_credit_type',
                   'submission_of_application', 'lump_sum_payment']
    x = pd.get_dummies(x, columns=categoricas, drop_first=True)

    # Padronizar os dados
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Treinar o modelo KMeans
    kmeans = KMeans(n_clusters=13, random_state=42)
    kmeans.fit(x_scaled)
    
    # Adicionar os clusters ao dataframe
    df_cluster['cluster'] = kmeans.labels_

    # Salvar o modelo em um arquivo pickle
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

# Função para carregar o modelo KMeans
def load_model():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    return kmeans_model

# Função para realizar a predição do cluster
def predict_cluster(data):
    global df_cluster
    
    kmeans_model = load_model()
    
    df = pd.DataFrame(data, index=[0])
    df = pd.get_dummies(df)
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    cluster = kmeans_model.predict(df_scaled)[0]
    
    # Mapear o número do cluster para o label correspondente
    labels = {
        0: 'Baixo Risco',
        1: 'Risco Moderado 1',
        2: 'Risco Moderado 2',
        3: 'Risco Moderado-Alto',
        4: 'Alto Risco 1',
        5: 'Muito Alto Risco',
        6: 'Risco Moderado-Alto 1',
        7: 'Risco Moderado 2',
        8: 'Alto Risco 2',
        9: 'Muito Alto Risco 1',
        10: 'Risco Moderado-Baixo',
        11: 'Risco Moderado 3',
        12: 'Muito Alto Risco 2'
    }
    label = labels[cluster]
    
    # Calcular a propensão média de default para o grupo do cluster
    propensao_default = df_cluster.groupby('cluster')['Status'].mean().loc[cluster]
    
    return {
        'cluster': int(cluster),  # Convertendo para int
        'label': label,
        'propensao_default': propensao_default
    }

app = Flask(__name__)

# Definir a rota para receber as solicitações
@app.route('/predict_clusterization', methods=['POST'])
def predict():
    data = request.json
    result = predict_cluster(data)
    return jsonify(result)

if __name__ == '__main__':
    # Treinar o modelo e iniciar a API Flask
    train_model()
    app.run(debug=True)
