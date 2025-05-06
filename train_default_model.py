import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import joblib

# Load PIMA dataset
df = pd.read_csv("pima_diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Buat model MLP di TensorFlow
model = Sequential([
    Dense(16, input_shape=(X_scaled.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')   # Output layer sigmoid untuk klasifikasi biner
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Latih model
model.fit(X_scaled, y, epochs=50, batch_size=16, verbose=1)

# Simpan model TensorFlow (.h5) dan scaler (.pkl)
model.save("pretrained/mlp_model_tf.h5")
joblib.dump({"scaler": scaler, "features": X.columns.tolist()}, "pretrained/mlp_model_meta.pkl")

print("Model TensorFlow berhasil disimpan sebagai mlp_model_tf.h5 dan mlp_model_meta.pkl")
