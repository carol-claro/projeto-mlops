from flask import Flask, request, jsonify
from model import predict_clusterization, predict_logistic_regression

app = Flask(__name__)

@app.route('/predict_clusterization', methods=['POST'])
def predict_clusterization_route():
    return jsonify(predict_clusterization())

@app.route('/predict_logistic_regression', methods=['POST'])
def predict_logistic_regression_route():
    return jsonify(predict_logistic_regression())

if __name__ == '__main__':
    app.run(debug=True)
