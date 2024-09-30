# prototipo_api.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Cargar el modelo RL-AD entrenado y el escalador
model = joblib.load('modelo_rl_ad.pkl')

# Inicializar Flask
app = Flask(__name__)

# Cargar el escalador guardado (o reentrenar si es necesario)
scaler = StandardScaler()

# Definir un endpoint para recibir datos y hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en la solicitud (en formato JSON)
        data = request.json
        # Convertir los datos a DataFrame y preprocesar
        features = pd.DataFrame([data['features']])

        # Realizar la misma transformación que en el entrenamiento
        scaled_features = scaler.fit_transform(features)

        # Realizar predicción con el modelo cargado
        prediction = model.predict(scaled_features)

        # Formatear la respuesta
        response = {
            'prediction': int(prediction[0]),
            'description': 'Anomalía detectada' if prediction[0] == 1 else 'Tráfico normal'
        }
        return jsonify(response)

    except Exception as e:
        # Manejo de errores
        return jsonify({'error': str(e)}), 400

# Endpoint para probar que el servidor está activo
@app.route('/ping', methods=['GET'])
def ping():
    return "API activa y lista para recibir predicciones.", 200

# Ejecutar la API
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
