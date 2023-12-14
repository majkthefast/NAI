"""
Sieć Neuronowa do Rozpoznawania Ubrań na Zbiorze Fashion-MNIST

Autor: [Twoje imię]
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Załaduj dane Fashion-MNIST
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalizacja pikseli do zakresu [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Zakoduj etykiety kategorialnie
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Stwórz model CNN
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
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
