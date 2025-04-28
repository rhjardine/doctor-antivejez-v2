import tensorflow as tf
import numpy as np

# Datos de ejemplo (series temporales de glucosa)
glucose_data = np.array([100, 105, 110, 108, 105, 102, 100, 98, 95, 93])
time_steps = 3

# Preparar datos para LSTM
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_dataset(glucose_data, time_steps)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Definir el modelo LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(time_steps, 1)),
    tf.keras.layers.Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(X, y, epochs=100, verbose=0)

# Guardar el modelo
model.save('glucose_prediction_model.h5')
print("Modelo de predicción de glucosa guardado como 'glucose_prediction_model.h5'")