"""
Sieć Neuronowa do Rozpoznawania Zwierząt na Zbiorze CIFAR-10

Autor: Mikołaj Prętki; Mikołaj Hołdakowski

Instrukcje Użycia:
Zainstaluj wymagane biblioteki, uruchamiając pip install tensorflow matplotlib.
Uruchom skrypt.
Referencje:
TensorFlow: https://www.tensorflow.org/
matplotlib: https://matplotlib.org/
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Załaduj dane CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalizacja pikseli do zakresu [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Zakoduj etykiety kategorialnie
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Stwórz model CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Wyświetlenie dokładności trenowania i walidacji
plt.plot(history.history['accuracy'], label='Dokładność trenowania')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacji')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()
plt.show()
