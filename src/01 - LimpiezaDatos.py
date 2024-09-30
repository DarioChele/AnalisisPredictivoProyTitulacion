# LimpiezaDatos.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar los datasets generados
df_normal = pd.read_csv('trafico_normal.csv')
df_high_latency = pd.read_csv('trafico_alta_latencia.csv')
df_ddos = pd.read_csv('trafico_ddos.csv')

# Función para limpiar datos
def clean_data(df):
    # Eliminar registros con valores nulos
    df = df.dropna()
    # Filtrar outliers en RTT y Size usando percentiles
    rtt_lower, rtt_upper = df['rtt'].quantile([0.01, 0.99])
    size_lower, size_upper = df['size'].quantile([0.01, 0.99])
    df = df[(df['rtt'] >= rtt_lower) & (df['rtt'] <= rtt_upper)]
    df = df[(df['size'] >= size_lower) & (df['size'] <= size_upper)]
    return df

# Aplicar limpieza a cada dataset
df_normal_clean = clean_data(df_normal)
df_high_latency_clean = clean_data(df_high_latency)
df_ddos_clean = clean_data(df_ddos)

# Función para normalizar datos
def normalize_data(df):
    scaler = MinMaxScaler()
    df[['rtt', 'size']] = scaler.fit_transform(df[['rtt', 'size']])
    return df

# Aplicar normalización a cada dataset limpio
df_normal_normalized = normalize_data(df_normal_clean)
df_high_latency_normalized = normalize_data(df_high_latency_clean)
df_ddos_normalized = normalize_data(df_ddos_clean)

# Función para agregar nuevas características
def feature_engineering(df):
    # Agregar la columna de fluctuación de latencia como la diferencia de RTT entre paquetes consecutivos
    df['latency_fluctuation'] = df['rtt'].diff().fillna(0)
    # Crear una columna que represente la tasa de paquetes por segundo, simulando un análisis en tiempo real
    df['packets_per_sec'] = 1 / df['rtt'].replace(0, 1)  # Evitar divisiones por cero
    return df

# Aplicar ingeniería de características a cada dataset normalizado
df_normal_features = feature_engineering(df_normal_normalized)
df_high_latency_features = feature_engineering(df_high_latency_normalized)
df_ddos_features = feature_engineering(df_ddos_normalized)

# Etiquetar los datasets
df_normal_features['label'] = 0  # Tráfico normal
df_high_latency_features['label'] = 1  # Alta latencia
df_ddos_features['label'] = 1  # DDoS simulado

# Combinar todos los datasets etiquetados en uno solo para el entrenamiento
df_combined = pd.concat([df_normal_features, df_high_latency_features, df_ddos_features], ignore_index=True)

# Guardar el dataset combinado para su uso en el entrenamiento del modelo
df_combined.to_csv('dataset_combinado.csv', index=False)
print("Datos preparados y guardados en 'dataset_combinado.csv'")
