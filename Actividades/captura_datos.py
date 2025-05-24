import cv2
import os

# Cambia esta letra por la que estés capturando
letra = 'C'

# Ruta donde se guardarán las imágenes
ruta_guardado = f'C:/Users/guapo/OneDrive/Escritorio/proyecto_sign_language/dataset/{letra}'
if not os.path.exists(ruta_guardado):
    os.makedirs(ruta_guardado)

# Inicializar webcam
camara = cv2.VideoCapture(0)
contador = 0
max_imagenes = 250  # puedes cambiar esto

while True:
    ret, frame = camara.read()
    if not ret:
        break

    # Mostrar el marco capturado
    cv2.putText(frame, f'Letra: {letra} | Imagenes: {contador}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Captura de Imagenes - Presiona "s" para guardar, "q" para salir', frame)

    # Leer teclado
    tecla = cv2.waitKey(1)

    # Si presiona 's' guarda imagen
    if tecla == ord('s') and contador < max_imagenes:
        nombre_imagen = f'{ruta_guardado}/{contador}.jpg'
        cv2.imwrite(nombre_imagen, frame)
        contador += 1
        print(f'[+] Imagen guardada: {nombre_imagen}')

    # Salir con 'q'
    elif tecla == ord('q'):
        break

# Liberar recursos
camara.release()
cv2.destroyAllWindows()
