# modelo_predictivo_combinado.py (Ajustado)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# Cargar el dataset combinado
df = pd.read_csv('dataset_combinado.csv')

# Eliminar columnas no numéricas que no son relevantes para el modelo
df = df.drop(['src_ip', 'dst_ip', 'protocol'], axis=1, errors='ignore')

# Separar características y etiquetas
X = df.drop(['label'], axis=1)  # Características
y = df['label']  # Etiquetas (0: normal, 1: anómalo)

# Dividir los datos en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Función para calcular el tiempo promedio de predicción
def calculate_prediction_time(model, X_test):
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    return (end_time - start_time) / len(X_test)

# Lista para almacenar los resultados
results = []

# **Desarrollo del Modelo Combinado RL-AD**
print("     --> Iniciando modelos Regresion Logistica y Arboles de Decision en un Voting Classifier...")
# Crear y combinar Regresión Logística y Árboles de Decisión en un Voting Classifier
model_logistic = LogisticRegression(max_iter=1000, random_state=42)
model_tree = DecisionTreeClassifier(random_state=42)

# Combinación de los modelos en un ensemble (voting='soft' pondera las predicciones)
model_combined = VotingClassifier(estimators=[('RL', model_logistic), ('AD', model_tree)], voting='soft')

# Entrenar el modelo combinado con el conjunto de entrenamiento
model_combined.fit(X_train, y_train)

# Validar el modelo combinado con el conjunto de validación
y_val_pred_combined = model_combined.predict(X_val)

# Evaluar el modelo combinado con el conjunto de prueba
y_test_pred_combined = model_combined.predict(X_test)

# Métricas del modelo combinado
val_accuracy = accuracy_score(y_val, y_val_pred_combined)
test_accuracy = accuracy_score(y_test, y_test_pred_combined)
conf_matrix = confusion_matrix(y_test, y_test_pred_combined)
false_positives = conf_matrix[0][1] / np.sum(conf_matrix[0]) * 100
false_negatives = conf_matrix[1][0] / np.sum(conf_matrix[1]) * 100
prediction_time_avg = calculate_prediction_time(model_combined, X_test)

# Guardar los resultados del modelo combinado RL-AD
print("     --> Guardando resultados de RL y AD.....")
results.append(['RL-AD (Combinado)', prediction_time_avg, test_accuracy * 100, false_positives, false_negatives])

# **Random Forest y XGBoost (Opcionales para comparación)**
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Modelo Random Forest
print("     --> Iniciando modelo Random Forest...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_test_pred_rf = model_rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred_rf)
conf_matrix = confusion_matrix(y_test, y_test_pred_rf)
false_positives = conf_matrix[0][1] / np.sum(conf_matrix[0]) * 100
false_negatives = conf_matrix[1][0] / np.sum(conf_matrix[1]) * 100
prediction_time_avg = calculate_prediction_time(model_rf, X_test)
print("     --> Guardando resultados de Random forest...")
results.append(['Random Forest', prediction_time_avg, test_accuracy * 100, false_positives, false_negatives])

# Modelo XGBoost
print("     --> Iniciando modelo XGBoost...")
model_xgb = xgb.XGBClassifier(eval_metric='logloss')
model_xgb.fit(X_train, y_train)
y_test_pred_xgb = model_xgb.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred_xgb)
conf_matrix = confusion_matrix(y_test, y_test_pred_xgb)
false_positives = conf_matrix[0][1] / np.sum(conf_matrix[0]) * 100
false_negatives = conf_matrix[1][0] / np.sum(conf_matrix[1]) * 100
prediction_time_avg = calculate_prediction_time(model_xgb, X_test)
print("     --> Guardando resultados de XGBoost...")
results.append(['XGBoost', prediction_time_avg, test_accuracy * 100, false_positives, false_negatives])

# Crear un DataFrame con los resultados y guardarlos en un archivo CSV
df_results = pd.DataFrame(results, columns=['Modelo', 'Tiempo Prediccion AVG (seg)', 'Precision (%)', 'Falsos Positivos (%)', 'Falsos Negativos (%)'])
df_results.to_csv('resultados_modelos.csv', index=False)

print("Resultados guardados en 'resultados_modelos.csv'")
