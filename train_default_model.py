import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

# Load PIMA dataset
df = pd.read_csv("pima_diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, learning_rate_init=0.001, random_state=42)
model.fit(X_scaled, y)

joblib.dump({"model": model, "scaler": scaler, "features": X.columns.tolist()}, "mlp_model.pkl")
print("Model default berhasil disimpan sebagai mlp_model.pkl")
