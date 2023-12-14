"""
Sieć Neuronowa do Przewidywania Jakości Wina

Autor: [Twoje imię]
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Odczytaj dane z pliku
file_path = 'winequality-white.csv'
wine_data = pd.read_csv(file_path, delimiter=';')

# Podziel dane na wejścia (X) i etykiety (y)
X = wine_data.drop('quality', axis=1)
y = wine_data['quality'].apply(lambda x: 1 if x > 5 else 0)  # Binarna etykieta: 1 - dobre, 0 - złe

# Podziel dane na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standaryzacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tworzenie modelu sieci neuronowej
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trening modelu
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Ocena modelu na zestawie testowym
y_pred_probs = model.predict(X_test_scaled)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

# Metryki oceny modelu
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Wyświetlenie wyników
print(f"Dokładność: {accuracy}")
print("\nMacierz Pomyłek:")
print(conf_matrix)
print("\nRaport Klasyfikacyjny:")
print(class_report)
