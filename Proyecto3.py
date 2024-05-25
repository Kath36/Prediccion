import pandas as pd
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

app = Flask(__name__)

# Leer y preparar los datos (ejemplo simplificado)
data = pd.read_csv('data.csv')
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = data['target']

# Normalizar los datos
escalador = preprocessing.MinMaxScaler()
XScalado = escalador.fit_transform(X)

# Entrenar el modelo
regresion = LogisticRegression(max_iter=1000)
regresion.fit(XScalado, y)

@app.route('/')
def home():
    return "Modelo de Predicción de Enfermedades Cardíacas"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    df_scaled = escalador.transform(df)
    prediction = regresion.predict(df_scaled)

    return jsonify({
        'prediction': 'enfermo' if prediction[0] == 1 else 'sano'
    })

if __name__ == '__main__':
    app.run(debug=True)
