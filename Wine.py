"""
Porównanie dwóch różnych rozmiarów sieci neuronowych w zadaniu klasyfikacji jakości wina.
Autor: Mikołaj Prętki; Mikołaj Hołdakowski
Instrukcja użycia: Uruchom ten skrypt w środowisku obsługującym Python i TensorFlow.
Upewnij się, że masz zainstalowane wymagane biblioteki (pandas, tensorflow, scikit-learn).
Uruchom skrypt.
Referencje:
TensorFlow: https://www.tensorflow.org/
matplotlib: https://matplotlib.org/
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Wczytaj dane
file_path = 'winequality-white.csv'
first_line = pd.read_csv(file_path, delimiter=';', nrows=0).columns.tolist()
column_names = [col.strip('"') for col in first_line]
X = pd.read_csv(file_path, delimiter=';', skiprows=1, names=column_names)

# Przygotuj dane
y = X['quality'].apply(lambda x: 1 if x > 5 else 0)
X = X.drop(columns='quality')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
y_train_binary = tf.keras.utils.to_categorical(y_train, 2)
y_test_binary = tf.keras.utils.to_categorical(y_test, 2)

# Sieć Neuronowa o Mniejszym Rozmiarze
model_small = tf.keras.models.Sequential([
    # Warstwa gęsto połączona z 32 neuronami i funkcją aktywacji ReLU
    tf.keras.layers.Dense(32, activation='relu', input_shape=(len(column_names)-1,)),
    # Warstwa gęsto połączona z 16 neuronami i funkcją aktywacji ReLU
    tf.keras.layers.Dense(16, activation='relu'),
    # Warstwa wyjściowa z 2 neuronami i funkcją aktywacji softmax (bo mamy dwie klasy)
    tf.keras.layers.Dense(2, activation='softmax')
])

# Kompilacja modelu
model_small.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model_small.fit(X_train, y_train_binary, epochs=10, batch_size=32, validation_split=0.2)

# Ocenianie modelu o mniejszym rozmiarze
y_pred_small = model_small.predict(X_test)
y_pred_small_binary = y_pred_small.argmax(axis=1)  # Konwersja kodowania one-hot na etykiety binarne
accuracy_small = accuracy_score(y_test, y_pred_small_binary)
print(f"Accuracy for Smaller Model: {accuracy_small}")

# Sieć Neuronowa o Większym Rozmiarze
model_large = tf.keras.models.Sequential([
    # Warstwa gęsto połączona z 128 neuronami i funkcją aktywacji ReLU
    tf.keras.layers.Dense(128, activation='relu', input_shape=(len(column_names)-1,)),
    # Warstwa gęsto połączona z 64 neuronami i funkcją aktywacji ReLU
    tf.keras.layers.Dense(64, activation='relu'),
    # Warstwa gęsto połączona z 32 neuronami i funkcją aktywacji ReLU
    tf.keras.layers.Dense(32, activation='relu'),
    # Warstwa wyjściowa z 2 neuronami i funkcją aktywacji softmax (bo mamy dwie klasy)
    tf.keras.layers.Dense(2, activation='softmax')
])

# Kompilacja modelu
model_large.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model_large.fit(X_train, y_train_binary, epochs=10, batch_size=32, validation_split=0.2)

# Ocenianie modelu o większym rozmiarze
y_pred_large = model_large.predict(X_test)
y_pred_large_binary = y_pred_large.argmax(axis=1)  # Konwersja kodowania one-hot na etykiety binarne
accuracy_large = accuracy_score(y_test, y_pred_large_binary)
print(f"Accuracy for Larger Model: {accuracy_large}")
