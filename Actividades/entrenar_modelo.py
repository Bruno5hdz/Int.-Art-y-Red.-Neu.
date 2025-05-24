import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 64
RUTA_DATASET = 'dataset/'
CLASES = os.listdir(RUTA_DATASET)
CLASES.sort()

X = []
y = []

for etiqueta, clase in enumerate(CLASES):
    ruta_clase = os.path.join(RUTA_DATASET, clase)
    for img_nombre in os.listdir(ruta_clase):
        img_path = os.path.join(ruta_clase, img_nombre)
        imagen = cv2.imread(img_path)
        if imagen is not None:
            imagen = cv2.resize(imagen, (IMG_SIZE, IMG_SIZE))
            imagen = imagen / 255.0
            X.append(imagen)
            y.append(etiqueta)

X = np.array(X)
y = np.array(y)

print(f"Total de imágenes cargadas: {len(X)}")
print(f"Clases detectadas: {CLASES}")

y_cat = to_categorical(y, num_classes=len(CLASES))
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

modelo = Sequential()
modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
modelo.add(MaxPooling2D((2, 2)))
modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(MaxPooling2D((2, 2)))
modelo.add(Flatten())
modelo.add(Dense(64, activation='relu'))
modelo.add(Dense(len(CLASES), activation='softmax'))

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Entrenando modelo...")
historial = modelo.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
)
print("Entrenamiento finalizado.")

modelo.save('modelo_senas.h5')
print("Modelo guardado como 'modelo_senas.h5'")

# Graficar métricas
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Precisión entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Precisión validación')
plt.title('Precisión por época')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Pérdida entrenamiento')
plt.plot(historial.history['val_loss'], label='Pérdida validación')
plt.title('Pérdida por época')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
