# script_entrenar_modelo.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib  # Para guardar y cargar el modelo entrenado
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# Función para cargar y preprocesar los datos
def cargar_datos():
    print("Cargando y preprocesando el dataset...")
    df = pd.read_csv('dataset_combinado.csv')
    df = df.drop(['src_ip', 'dst_ip', 'protocol'], axis=1, errors='ignore')
    X = df.drop(['label'], axis=1)
    y = df['label']
    return X, y

# Función para dividir los datos y aplicar SMOTE
def preparar_datos(X, y):
    print("Dividiendo los datos en conjuntos de entrenamiento, validación y prueba...")
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

# Función para entrenar y guardar el modelo combinado RL-AD
def entrenar_y_guardar_modelo(X_train, y_train):
    print("\nEntrenando el modelo combinado RL-AD...")

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

    # Guardar el modelo entrenado
    joblib.dump(best_combined, 'modelo_rl_ad.pkl')
    print("Modelo RL-AD guardado como 'modelo_rl_ad.pkl'.")

    return best_combined

# Función principal para ejecutar el entrenamiento y guardado del modelo
def main():
    # Cargar y preparar los datos
    X, y = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preparar_datos(X, y)

    # Entrenar y guardar el modelo RL-AD
    entrenar_y_guardar_modelo(X_train, y_train)

# Ejecutar el script de entrenamiento
if __name__ == "__main__":
    main()
