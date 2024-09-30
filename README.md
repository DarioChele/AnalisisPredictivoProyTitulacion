# Título del Proyecto
Análisis Predictivo para la Detección de Latencia en Redes Corporativas

# Descripción del Proyecto
Este proyecto tiene como objetivo implementar un sistema predictivo para la detección de latencias anómalas en redes corporativas, utilizando modelos de machine learning como Regresión Logística, Árboles de Decisión, Random Forest y XGBoost. El modelo final, RL-AD, fue seleccionado por su precisión, baja tasa de falsos positivos y mejores tiempos de predicción, asegurando la detección en tiempo real de posibles problemas de latencia en redes.

# Características Principales
* Detección en Tiempo Real: Modelos entrenados para analizar y predecir anomalías en el tráfico de red con tiempos de respuesta inferiores a 500 ms.
* Modelos de Machine Learning: Implementación y evaluación de varios modelos predictivos como RL-AD, Random Forest y XGBoost.
* Optimización de Rendimiento: Ajuste y selección de los modelos más precisos y rápidos para la detección de latencias en redes corporativas.
* Monitoreo Continuo: Integración con una API para recibir datos de tráfico de red en tiempo real.
# Requisitos
Los siguientes paquetes y herramientas deben estar instalados para ejecutar el proyecto:

* Python 3.7+
* Paquetes Python:
  - scikit-learn
  - pandas
  - numpy
  - xgboost
  - matplotlib
  - flask 
  - joblib (para guardar y cargar los modelos)
  - 
Instala los requisitos ejecutando:
```bash
pip install -r requirements.txt
````
# Estructura del Repositorio
```plaintext
├── data/                         # Directorio para almacenar los conjuntos de datos recopilados para el entrenamiento 
├── src/                          # Código fuente del proyecto
│   ├── 01 - LimpiezaDatos.py     # Código para preprocesamiento de datos
│   ├── 02 - ModeloPredictivo.py  # Entrenamiento y evaluación de los modelos junto con sus respectivas métricas
│   ├── 03 - Prototipo.py         # Implementación de la API para monitoreo en tiempo real
├── README.md                     # Archivo que estás leyendo ahora
├── requirements.txt              # Lista de dependencias
└── modelo_rl_ad.pkl              # Modelo entrenado RL-AD (opcional)
```
# Instalación y Ejecución
1. Clonar el Repositorio:
```bash
git clone https://github.com/DarioChele/AnalisisPredictivoProyTitulacion.git

```
2. Instalar Dependencias: Ejecuta el siguiente comando para instalar las dependencias:
```bash
pip install -r requirements.txt
```
3. Entrenar los Modelos: Para entrenar los modelos, ejecuta el script model_training.py:
```bash
python src/02 - ModeloPredictivo.py
```
4. Cargar el Modelo y Ejecutar la API: Si deseas ejecutar el motor predictivo en tiempo real, ejecuta el siguiente comando:
```bash
python src/03 - Prototipo.py
```
# Uso del Prototipo
Una API fue implementada para recibir tráfico de red en tiempo real y evaluar la latencia del mismo. Los datos pueden enviarse en formato JSON a la URL predeterminada (http://localhost:5000/predict).

Ejemplo de petición:
```json
{
  "features": [30.5, 850, 10, 55, 1]
}
```
El array <mark>features</mark> contiene los valores de las características del tráfico de red que se enviarán al modelo para hacer una predicción. Cada número en la lista representa una característica del tráfico de red. A continuación, te explico qué representa cada valor:
- RTT (ms) (30.5): El Round Trip Time es el tiempo que tarda un paquete en viajar desde el origen hasta el destino y regresar. En este caso, 30.5 milisegundos.
- Tamaño del Paquete (bytes) (850): Tamaño del paquete de datos en bytes que se está enviando o recibiendo. En este ejemplo, el tamaño del paquete es de 850 bytes.
- Conexiones Simultáneas (10): Número de conexiones simultáneas activas en el momento del análisis. Aquí tenemos 10 conexiones simultáneas.
- Uso del Ancho de Banda (%) (55): Porcentaje de ancho de banda que está siendo utilizado en ese momento. En este caso, el 55% del ancho de banda está ocupado.
- Retransmisiones (1): Número de veces que un paquete ha sido retransmitido debido a errores o pérdida de datos. Aquí se ha registrado 1 retransmisión.

Estos datos se utilizan como entrada para el modelo, y la API devolverá si el tráfico es "normal" o "anómalo" según el análisis

# Posibles Respuestas de la API
La API puede devolver uno de los siguientes resultados, dependiendo de la predicción del modelo:
- "Tráfico normal": El tráfico de red es normal según el análisis del modelo.
- "Anomalía detectada": El tráfico de red presenta comportamientos anómalos que podrían estar relacionados con problemas de latencia o ataques


# Usar la API con POSTMAN
Si ya tienes la API ejecutándose localmente, puedes usar POSTMAN para probarla enviando datos de tráfico de red.

## Pasos para Probar la API empleando POSTMAN 
1. Asegúrate de que la API esté ejecutándose:
  * Primero, inicia la API en tu máquina local usando el siguiente comando (asegúrate de estar en el directorio correcto):
```bash
python src/api.py
```
  * La API debería estar escuchando en http://localhost:5000/predict.
2. Crear una Nueva Solicitud en POSTMAN:
  * Haz clic en "New" y selecciona "Request".
  * Nombra tu solicitud, por ejemplo, "Prueba de API de Predicción de Latencia", y selecciona una colección o carpeta para guardarla.
  * Haz clic en "Save to".
3. Configurar el Tipo de Solicitud:
  * Selecciona POST como tipo de solicitud HTTP.
  * En la barra de URL, introduce <mark>http://localhost:5000/predict</mark> (o la dirección de tu API si está en otro servidor).
4. Configurar el Cuerpo de la Solicitud (Body):
  * Ve a la pestaña Body y selecciona raw.
  * En el menú desplegable que aparece a la derecha de "raw", selecciona JSON.
5. Introducir el JSON de la Solicitud:
  * Escribe o pega el JSON que quieres enviar a la API en el área de texto. Un ejemplo de solicitud JSON sería:
```json
{
  "features": [30.5, 850, 10, 55, 1]
}
```
  * Este JSON contiene las características del tráfico de red (RTT, tamaño de paquete, etc.) que el modelo evaluará.
6. Enviar la Solicitud:
  * Haz clic en Send para enviar la solicitud POST.
  * La API evaluará los datos y devolverá una respuesta JSON indicando si el tráfico es normal o anómalo.
7. Ver la Respuesta:
  * POSTMAN mostrará la respuesta de la API en el panel de "Response" abajo. Un ejemplo de respuesta podría ser:
```json
{
  "prediction": "Anomalía detectada"
}
```
  * En este ejemplo, la API está indicando que el tráfico evaluado es anómalo.

# Redirigir tráfico hacia la API con pyshark
Si deseas evaluar el comportamiento de la API con tráfico de red en tiempo real, puedes utilizar herramientas como pyshark para capturar paquetes directamente desde una interfaz de red y enviarlos a la API.
Requisitos:
- pyshark: Herramienta que permite capturar y analizar paquetes de red en tiempo real. Puedes instalarla ejecutando:
```bash
pip install pyshark
```
Con pyshark, puedes capturar tráfico de red en tiempo real y procesarlo automáticamente para enviarlo a la API para su evaluación, este enfoque es útil para realizar pruebas en escenarios controlados o simulados de tráfico de red, lo que permite verificar la capacidad de la API para detectar latencias anómalas en entornos dinámicos.


