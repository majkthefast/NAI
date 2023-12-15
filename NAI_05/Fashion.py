"""
Sieć Neuronowa do Rozpoznawania Ubrań na Zbiorze Fashion-MNIST

Autor: Mikołaj Prętki; Mikołaj Hołdakowski

Instrukcje Użycia:
Zainstaluj wymagane biblioteki, uruchamiając pip install tensorflow matplotlib.
Uruchom skrypt.
Referencje:
TensorFlow: https://www.tensorflow.org/
matplotlib: https://matplotlib.org/
"""
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
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

# Ocenianie modelu na zestawie testowym
y_pred_probs = model.predict(test_images)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(test_labels, axis=1)

# Tworzenie macierzy pomyłek
conf_matrix = confusion_matrix(y_true, y_pred)

# Wyświetlanie macierzy pomyłek za pomocą Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
