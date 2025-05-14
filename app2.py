import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
import io
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MLP Diabetes App", layout="wide")

# Fungsi untuk melatih model menggunakan TensorFlow
def train_model_tf(X, y, hidden_layers, activations, max_iter, learning_rate):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(X.shape[1],)))
    for units, act in zip(hidden_layers, activations):
        model.add(tf.keras.layers.Dense(units, activation=act))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(X_train_scaled, y_train,
                        validation_data=(X_test_scaled, y_test),
                        epochs=max_iter,
                        verbose=0)

    y_pred_probs = model.predict(X_test_scaled)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    return model, scaler, history, y_test, y_pred

# Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["Training Model", "Coba Model"])

if page == "Training Model":
    st.title("Training Model Diabetes (TensorFlow MLP)")

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
            layer_config = st.text_input("Konfigurasi Hidden Layers (contoh: 64-relu,32-tanh)", value="64-relu,32-relu")
            max_iter = st.number_input("Jumlah Epoch", min_value=1, value=50)
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format="%.6f")

            if st.button("Mulai Training"):
                with st.spinner("Training model..."):
                    hidden_layers = []
                    activations = []
                    for layer in layer_config.split(","):
                        units, act = layer.strip().split("-")
                        hidden_layers.append(int(units))
                        activations.append(act)

                    model, scaler, history, y_test, y_pred = train_model_tf(X, y, hidden_layers, activations, max_iter, learning_rate)

                    # Simpan ke session_state
                    st.session_state["model"] = model
                    st.session_state["scaler"] = scaler
                    st.session_state["feature_columns"] = feature_columns
                    st.session_state["history"] = history
                    st.session_state["y_test"] = y_test
                    st.session_state["y_pred"] = y_pred

                st.success("Training selesai!")

    # Menampilkan hasil training jika sudah ada di session
    if "model" in st.session_state:
        history = st.session_state["history"]
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]

        # Plot loss & accuracy
        hist_df = pd.DataFrame(history.history)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(hist_df["loss"], label="Train Loss")
        ax[0].plot(hist_df["val_loss"], label="Val Loss")
        ax[0].set_title("Loss Curve")
        ax[0].legend()

        ax[1].plot(hist_df["accuracy"], label="Train Acc")
        ax[1].plot(hist_df["val_accuracy"], label="Val Acc")
        ax[1].set_title("Accuracy Curve")
        ax[1].legend()

        st.pyplot(fig)

        st.write(f"**Akurasi Akhir (Testing):** {history.history['val_accuracy'][-1]:.4f}")
        st.write(f"**Loss Akhir (Testing):** {history.history['val_loss'][-1]:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        st.pyplot(fig_cm)

        # Hasil prediksi vs label asli
        st.subheader("Hasil Prediksi vs Label Asli (Data Uji)")
        result_df = pd.DataFrame({"Label Asli": y_test, "Prediksi": y_pred}).reset_index(drop=True)
        st.dataframe(result_df)

        # Download model dan scaler
        st.subheader("Download Model dan Scaler")

        # Simpan model ke file sementara
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_model:
            st.session_state["model"].save(tmp_model.name)
            tmp_model.seek(0)
            model_data = tmp_model.read()

        st.download_button("Download Model (.h5)", model_data, file_name="mlp_tf_model.h5")

        # Simpan scaler + features ke buffer
        scaler_buffer = io.BytesIO()
        joblib.dump({
            "scaler": st.session_state["scaler"],
            "features": st.session_state["feature_columns"]
        }, scaler_buffer)
        scaler_buffer.seek(0)

        st.download_button("Download Scaler + Features (.pkl)", scaler_buffer, file_name="scaler_features.pkl")

elif page == "Coba Model":
    st.title("Coba Model yang Sudah Dilatih")

    uploaded_model = st.file_uploader("Upload Model (*.h5)", type=["h5"])
    uploaded_scaler = st.file_uploader("Upload Scaler + Features (*.pkl)", type=["pkl"])

    # Gunakan model dan scaler default jika user tidak upload
    model_path = "pretrained/mlp_model_tf.h5"
    scaler_path = "pretrained/mlp_model_meta.pkl"

    try:
        if uploaded_model and uploaded_scaler:
            # Simpan uploaded_model ke file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_model_file:
                tmp_model_file.write(uploaded_model.read())
                tmp_model_file_path = tmp_model_file.name

            model = tf.keras.models.load_model(tmp_model_file_path)
            model_dict = joblib.load(uploaded_scaler)
            st.success("Model dan scaler berhasil dimuat dari upload!")
        else:
            model = tf.keras.models.load_model(model_path)
            with open(scaler_path, "rb") as f:
                model_dict = joblib.load(f)
            st.info("Model dan scaler default berhasil dimuat!")

        scaler = model_dict["scaler"]
        features = model_dict["features"]

        st.subheader("Masukkan Data untuk Prediksi:")
        input_data = {}
        for feat in features:
            input_data[feat] = st.number_input(f"{feat}:", value=0.0)

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction_prob = model.predict(input_scaled)[0][0]
        prediction = int(prediction_prob > 0.5)

        st.subheader("Hasil Prediksi:")
        st.write(f"Model memprediksi: **{prediction}** (Probabilitas: {prediction_prob:.4f})")

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
