import pyshark
import requests

# Configura la interfaz (usa el número o nombre que obtuviste con tshark -D)
INTERFAZ = 'Wi-Fi'  # O '1' si prefieres usar el número de la interfaz

# URL de la API que procesa los datos
API_URL = 'http://localhost:5000/predict'

# Función para procesar cada paquete capturado
def procesar_paquete(paquete):
    try:
        # Extraer el tamaño del paquete (en bytes)
        tamaño = int(paquete.length)

        # Simular datos adicionales que requiere la API
        rtt = 30.5  # Ejemplo de RTT en ms
        conexiones_simultaneas = 10
        uso_ancho_banda = 55  # Porcentaje de ancho de banda
        retransmisiones = 1

        # Preparar los datos en el formato que espera la API
        data = {
            "features": [tamaño, rtt, conexiones_simultaneas, uso_ancho_banda, retransmisiones]
        }

        # Enviar los datos a la API mediante POST
        respuesta = requests.post(API_URL, json=data)

        # Imprimir la respuesta de la API
        print(f"Respuesta de la API: {respuesta.json()}")

    except Exception as e:
        print(f"Error procesando el paquete: {e}")

# Iniciar la captura de paquetes en la interfaz especificada
print("Iniciando captura...")
capture = pyshark.LiveCapture(interface=INTERFAZ)

# Aplicar la función a cada paquete capturado
capture.apply_on_packets(procesar_paquete)