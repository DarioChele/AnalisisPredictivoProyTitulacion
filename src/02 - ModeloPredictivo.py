import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import time
import joblib  # Para guardar y cargar el modelo entrenado

# Funcion para cargar y preprocesar los datos
def cargar_datos():
    print("Cargando y preprocesando el dataset...")
    df = pd.read_csv('dataset_combinado.csv')
    df = df.drop(['src_ip', 'dst_ip', 'protocol'], axis=1, errors='ignore')
    X = df.drop(['label'], axis=1)
    y = df['label']
    return X, y

# Funcion para dividir los datos y aplicar SMOTE
def preparar_datos(X, y):
    print("Dividiendo los datos en conjuntos de entrenamiento, validacion y prueba...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    print("Aplicando SMOTE para balancear las clases...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("Estandarizando las características...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# Funcion para calcular el tiempo promedio de prediccion
def calculate_prediction_time(model, X_test):
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    return (end_time - start_time) / len(X_test)

# Funcion para entrenar y evaluar el modelo combinado RL-AD
def entrenar_rl_ad(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\nEntrenando el modelo combinado RL-AD...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import VotingClassifier

    # Definir los modelos base
    model_logistic = LogisticRegression(max_iter=1000, random_state=42)
    model_tree = DecisionTreeClassifier(random_state=42)
    model_combined = VotingClassifier(estimators=[('RL', model_logistic), ('AD', model_tree)], voting='soft')

    # Definir hiperparámetros para ajustar
    param_grid = {
        'RL__C': [0.1, 1, 10],
        'AD__max_depth': [None, 10, 20, 30],
    }

    # Búsqueda de los mejores hiperparámetros
    grid_combined = GridSearchCV(model_combined, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_combined.fit(X_train, y_train)
    best_combined = grid_combined.best_estimator_

    # Evaluacion del modelo
    y_test_pred_combined = best_combined.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred_combined)
    conf_matrix = confusion_matrix(y_test, y_test_pred_combined)
    false_positives = conf_matrix[0][1] / np.sum(conf_matrix[0]) * 100
    false_negatives = conf_matrix[1][0] / np.sum(conf_matrix[1]) * 100
    prediction_time_avg = calculate_prediction_time(best_combined, X_test)

    # Guardar el modelo
    joblib.dump(best_combined, 'modelo_rl_ad.pkl')
    print("Modelo RL-AD guardado como 'modelo_rl_ad.pkl'.")

    # Resultados del modelo
    resultados = ['RL-AD (Combinado)', prediction_time_avg, test_accuracy * 100, false_positives, false_negatives]
    print("\nResultados del modelo RL-AD:")
    print(f"Precision: {test_accuracy * 100:.2f}%")
    print(f"Falsos Positivos: {false_positives:.2f}%")
    print(f"Falsos Negativos: {false_negatives:.2f}%")
    print(f"Tiempo Promedio de Prediccion: {format(prediction_time_avg, '.10f')} segundos")

    return resultados

# Funcion principal para ejecutar el prototipo
def main():
    # Cargar y preparar los datos
    X, y = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preparar_datos(X, y)

    # Entrenar y evaluar el modelo RL-AD
    resultados_rl_ad = entrenar_rl_ad(X_train, y_train, X_val, y_val, X_test, y_test)

    # Guardar los resultados en CSV
    df_results = pd.DataFrame([resultados_rl_ad], columns=['Modelo', 'Tiempo Prediccion AVG (seg)', 'Precision (%)', 'Falsos Positivos (%)', 'Falsos Negativos (%)'])
    df_results.to_csv('resultados_modelo_rl_ad.csv', index=False)
    print("\nResultados guardados en 'resultados_modelo_rl_ad.csv'.")

# Ejecutar el prototipo
if __name__ == "__main__":
    main()
