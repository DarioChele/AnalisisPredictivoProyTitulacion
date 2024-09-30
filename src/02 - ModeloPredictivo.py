import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import time

# Cargar el dataset
df = pd.read_csv('dataset_combinado.csv')

# Eliminar columnas no relevantes
df = df.drop(['src_ip', 'dst_ip', 'protocol'], axis=1, errors='ignore')

# Separar características y etiquetas
X = df.drop(['label'], axis=1)
y = df['label']

# Dividir los datos en conjuntos de entrenamiento, validacion y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Aplicar SMOTE para manejar el desbalance de clases
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Funcion para calcular el tiempo promedio de prediccion
def calculate_prediction_time(model, X_test):
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    return (end_time - start_time) / len(X_test)

# Lista para almacenar los resultados
results = []

# Afinacion y evaluacion del modelo combinado RL-AD
print("     --> Iniciando modelos Regresion Logistica y Arboles de Decision en un Voting Classifier...")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# Definir los modelos base
model_logistic = LogisticRegression(max_iter=1000, random_state=42)
model_tree = DecisionTreeClassifier(random_state=42)

# Clasificador de votacion con votacion suave
model_combined = VotingClassifier(estimators=[('RL', model_logistic), ('AD', model_tree)], voting='soft')

# Definir hiperparámetros para ajustar
param_grid = {
    'RL__C': [0.1, 1, 10],
    'AD__max_depth': [None, 10, 20, 30],
}

# Realizar búsqueda en la cuadrícula
grid_combined = GridSearchCV(model_combined, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_combined.fit(X_train, y_train)

# Usar el mejor modelo encontrado
best_combined = grid_combined.best_estimator_

# Validacion y prueba
y_val_pred_combined = best_combined.predict(X_val)
y_test_pred_combined = best_combined.predict(X_test)

# Recopilar métricas
val_accuracy = accuracy_score(y_val, y_val_pred_combined)
test_accuracy = accuracy_score(y_test, y_test_pred_combined)
conf_matrix = confusion_matrix(y_test, y_test_pred_combined)
false_positives = conf_matrix[0][1] / np.sum(conf_matrix[0]) * 100
false_negatives = conf_matrix[1][0] / np.sum(conf_matrix[1]) * 100
prediction_time_avg = calculate_prediction_time(best_combined, X_test)

# Guardar los resultados
print("     --> Guardando resultados de RL y AD.....")
results.append(['RL-AD (Combinado)', prediction_time_avg, test_accuracy * 100, false_positives, false_negatives])

# Afinacion y evaluacion de Random Forest
print("     --> Iniciando modelo Random Forest...")
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
}

grid_rf = GridSearchCV(model_rf, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

y_test_pred_rf = best_rf.predict(X_test)

# Recopilar métricas
test_accuracy = accuracy_score(y_test, y_test_pred_rf)
conf_matrix = confusion_matrix(y_test, y_test_pred_rf)
false_positives = conf_matrix[0][1] / np.sum(conf_matrix[0]) * 100
false_negatives = conf_matrix[1][0] / np.sum(conf_matrix[1]) * 100
prediction_time_avg = calculate_prediction_time(best_rf, X_test)

print("     --> Guardando resultados de Random forest...")
results.append(['Random Forest', prediction_time_avg, test_accuracy * 100, false_positives, false_negatives])

# Afinacion y evaluacion de XGBoost
print("     --> Iniciando modelo XGBoost...")
import xgboost as xgb

model_xgb = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_xgb = GridSearchCV(model_xgb, param_grid_xgb, cv=3, scoring='accuracy', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_

y_test_pred_xgb = best_xgb.predict(X_test)

# Recopilar métricas
test_accuracy = accuracy_score(y_test, y_test_pred_xgb)
conf_matrix = confusion_matrix(y_test, y_test_pred_xgb)
false_positives = conf_matrix[0][1] / np.sum(conf_matrix[0]) * 100
false_negatives = conf_matrix[1][0] / np.sum(conf_matrix[1]) * 100
prediction_time_avg = calculate_prediction_time(best_xgb, X_test)

print("     --> Guardando resultados de XGBoost...")
results.append(['XGBoost', prediction_time_avg, test_accuracy * 100, false_positives, false_negatives])

# Guardar resultados en CSV
df_results = pd.DataFrame(results, columns=['Modelo', 'Tiempo Prediccion AVG (seg)', 'Precision (%)', 'Falsos Positivos (%)', 'Falsos Negativos (%)'])

# Convertir los números en notacion científica a un formato decimal legible
df_results['Tiempo Prediccion AVG (seg)'] = df_results['Tiempo Prediccion AVG (seg)'].apply(lambda x: format(x, '.10f'))
df_results['Precision (%)'] = df_results['Precision (%)'].apply(lambda x: format(x, '.2f'))
df_results['Falsos Positivos (%)'] = df_results['Falsos Positivos (%)'].apply(lambda x: format(x, '.2f'))
df_results['Falsos Negativos (%)'] = df_results['Falsos Negativos (%)'].apply(lambda x: format(x, '.2f'))

# Guardar los resultados en un archivo CSV
df_results.to_csv('resultados_modelos_ajustados.csv', index=False)
print("     -------------------------------------><------------------------------------- ")
print("     --> Resultados ajustados guardados en 'resultados_modelos_ajustados.csv'....")
print("     -------------------------------------><------------------------------------- ")
