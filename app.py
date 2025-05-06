import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MLP Diabetes App", layout="wide")

# Fungsi untuk melatih model
def train_model(X, y, hidden_layer_sizes, max_iter, learning_rate_init):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                          max_iter=max_iter,
                          learning_rate_init=learning_rate_init,
                          random_state=42)

    loss_curve = []
    acc_curve = []

    # Training manual (epoch-wise)
    for i in range(max_iter):
        model.partial_fit(X_train_scaled, y_train, classes=np.unique(y))
        loss_curve.append(model.loss_)
        y_pred = model.predict(X_test_scaled)
        acc_curve.append(accuracy_score(y_test, y_pred))

    return model, scaler, loss_curve, acc_curve, y_test, y_pred

# Halaman utama
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["Training Model", "Coba Model"])

if page == "Training Model":
    st.title("Training Model Diabetes (MLPClassifier)")

    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview Dataset:")
        st.dataframe(df.head())

        target_column = st.selectbox("Pilih Kolom Target", df.columns)
        feature_columns = st.multiselect("Pilih Kolom Fitur", [col for col in df.columns if col != target_column])

        if feature_columns and target_column:
            X = df[feature_columns]
            y = df[target_column]

            st.subheader("Parameter Model")
            hidden_layer = st.text_input("Hidden Layers (contoh: 100, atau 100,50 atau 50,100,100)", value="100")
            max_iter = st.number_input("Jumlah Epoch", min_value=1, value=50)
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format="%.6f")

            if st.button("Mulai Training"):
                with st.spinner("Training model..."):
                    hidden_layers = tuple(map(int, hidden_layer.split(",")))
                    model, scaler, loss_curve, acc_curve, y_test, y_pred = train_model(
                        X, y, hidden_layers, max_iter, learning_rate
                    )

                st.success("Training selesai!")

                # Plot loss & accuracy
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(loss_curve)
                ax[0].set_title("Loss Curve")
                ax[0].set_xlabel("Epoch")
                ax[0].set_ylabel("Loss")

                ax[1].plot(acc_curve)
                ax[1].set_title("Accuracy Curve")
                ax[1].set_xlabel("Epoch")
                ax[1].set_ylabel("Accuracy")

                st.pyplot(fig)
                
                st.subheader("Nilai Akurasi dan Loss Akhir")
                st.write(f"**Akurasi Akhir:** {acc_curve[-1]:.4f}")
                st.write(f"**Loss Akhir:** {loss_curve[-1]:.4f}")

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                st.subheader("Confusion Matrix")
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("True")
                st.pyplot(fig_cm)

                # Download model
                st.subheader("Download Model")
                joblib.dump({"model": model, "scaler": scaler, "features": feature_columns}, "mlp_model.pkl")
                with open("mlp_model.pkl", "rb") as f:
                    st.download_button("Download Model", f, file_name="mlp_model.pkl")

elif page == "Coba Model":
    st.title("Coba Model yang Sudah Dilatih")

    uploaded_model = st.file_uploader("Upload Model (*.pkl)", type=["pkl"])

    if uploaded_model:
        model_dict = joblib.load(uploaded_model)
        st.success("Model berhasil dimuat dari upload!")
    else:
        try:
            model_dict = joblib.load("mlp_model.pkl")  # file default
            st.info("Model bawaan digunakan karena tidak ada file upload.")
        except FileNotFoundError:
            st.error("Model default tidak ditemukan. Silakan upload model terlebih dahulu.")
            st.stop()

        model = model_dict["model"]
        scaler = model_dict["scaler"]
        features = model_dict["features"]

        st.success("Model berhasil dimuat!")

        st.subheader("Masukkan Data untuk Prediksi:")
        input_data = {}
        for feat in features:
            input_data[feat] = st.number_input(f"{feat}:", value=0.0)

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.subheader("Hasil Prediksi:")
        st.write(f"Model memprediksi: **{prediction}**")

