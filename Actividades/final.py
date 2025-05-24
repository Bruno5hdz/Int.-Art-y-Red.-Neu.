import cv2
import numpy as np
import pyautogui  # NUEVO: para presionar teclas
from tensorflow.keras.models import load_model
import mediapipe as mp
import time  # NUEVO: para evitar múltiples pulsaciones
import serial  # NUEVO: para comunicación con Arduino

# Parámetros
IMG_SIZE = 64
CLASES = sorted(['A', 'B', 'C'])  # Aquí tus clases
LETRA_OBJETIVO = 'C'  # <--- Cambia esta letra por la que quieras que active la tecla espacio

# Cargar modelo
modelo = load_model('modelo_senas.h5')

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# NUEVO: Conexión con Arduino
try:
    arduino = serial.Serial('COM9', 9600)  # Ajusta el puerto si es necesario
    time.sleep(2)  # Esperar a que el puerto se estabilice
    print("Conectado a Arduino.")
except Exception as e:
    print(f"Error al conectar con Arduino: {e}")
    arduino = None

def preprocesar(imagen):
    imagen = cv2.resize(imagen, (IMG_SIZE, IMG_SIZE))
    imagen = imagen / 255.0
    imagen = np.expand_dims(imagen, axis=0)
    return imagen

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

print("Presiona 'q' para salir.")

ultimo_tiempo = 0
espera = 2  # segundos entre pulsaciones para evitar múltiples envíos

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)
    
    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = frame.shape
            x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
            y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
            x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
            y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            mano_roi = frame[y_min:y_max, x_min:x_max]
            
            if mano_roi.size != 0:
                imagen_pre = preprocesar(mano_roi)
                pred = modelo.predict(imagen_pre)
                clase_idx = np.argmax(pred)
                prob = pred[0][clase_idx]
                letra_predicha = CLASES[clase_idx]

                texto = f"L: {letra_predicha} ({prob*100:.1f}%)"
                cv2.putText(frame, texto, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                tiempo_actual = time.time()
                
                # Enviar a Arduino si hay alta probabilidad y ha pasado tiempo suficiente
                if prob > 0.9 and (tiempo_actual - ultimo_tiempo > espera):
                    if arduino:
                        arduino.write(f"{letra_predicha}\n".encode())
                        print(f"Letra {letra_predicha} enviada a Arduino.")
                    
                    if letra_predicha == LETRA_OBJETIVO:
                        pyautogui.press('space')
                        print(f"Letra {LETRA_OBJETIVO} detectada. Se presionó ESPACIO.")

                    ultimo_tiempo = tiempo_actual
    
    cv2.imshow('Reconocimiento Lenguaje de Señas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if arduino:
    arduino.close()
cv2.destroyAllWindows()