import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# SET PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Stunting & Kanker Paru",
    page_icon="🏥",
    layout="wide"
)

# =========================
# LOAD MODELS (CACHE)
# =========================
@st.cache_resource
def load_models():
    model_stunting = joblib.load("knn_model.pkl")
    scaler_stunting = joblib.load("scaler.pkl")
    model_paru = joblib.load("model_paru_pas.pkl")
    scaler_paru = joblib.load("scaler_paru.pkl")
    return model_stunting, scaler_stunting, model_paru, scaler_paru

model_stunting, scaler_stunting, model_paru, scaler_paru = load_models()

# =========================
# LABEL MAPS
# =========================
label_map_stunting = {
    0: "Sangat Pendek",
    1: "Pendek",
    2: "Normal",
    3: "Tinggi"
}

label_map_paru = {
    0: "Tidak Terdeteksi Kanker",
    1: "Terdeteksi Kanker Paru-Paru"
}

# =========================
# FUNGSI REKOMENDASI STUNTING
# =========================
def get_stunting_recommendation(status):
    if status == "Sangat Pendek":
        return {
            "risk": "🔴 HIGH",
            "message": "Segera konsultasi ke dokter anak atau ahli gizi untuk pemeriksaan pertumbuhan dan pola makan balita."
        }
    elif status == "Pendek":
        return {
            "risk": "🟠 MODERATE",
            "message": "Perlu perhatian terhadap asupan gizi dan pemeriksaan rutin pertumbuhan anak."
        }
    elif status == "Normal":
        return {
            "risk": "🟢 LOW",
            "message": "Pertumbuhan balita normal. Tetap jaga pola makan sehat dan rutin cek pertumbuhan."
        }
    elif status == "Tinggi":
        return {
            "risk": "🟢 LOW",
            "message": "Tinggi badan di atas rata-rata. Tetap perhatikan keseimbangan nutrisi anak."
        }
    return {
        "risk": "🟡 MODERATE",
        "message": "Data tidak dikenali."
    }

# =========================
# FUNGSI REKOMENDASI PARU
# =========================
def get_paru_recommendation(prediction, confidence):
    if prediction == 1:
        if confidence >= 80:
            return "🔴 Segera konsultasi ke dokter spesialis paru."
        elif confidence >= 60:
            return "🟠 Disarankan pemeriksaan lanjutan."
        else:
            return "🟡 Konsultasi dengan dokter umum."
    else:
        if confidence >= 80:
            return "✅ Risiko rendah, tetap jaga kesehatan."
        elif confidence >= 60:
            return "✅ Perhatikan gejala jika berlanjut."
        else:
            return "🟡 Tetap lakukan pemeriksaan rutin."

def get_paru_risk_level(prediction, confidence):
    if prediction == 1:
        return "🔴 HIGH" if confidence >= 60 else "🟠 MODERATE"
    return "🟢 LOW" if confidence >= 60 else "🟡 MODERATE"

def calculate_confidence_paru(model, X_scaled):
    try:
        if hasattr(model, 'kneighbors'):
            k = model.n_neighbors
            distances, indices = model.kneighbors(X_scaled)
            neighbor_labels = model._y[indices[0]]
            prediction = model.predict(X_scaled)[0]
            same_label_count = np.sum(neighbor_labels == prediction)
            confidence = (same_label_count / k) * 100
            return round(confidence, 1)
        return 75.0
    except Exception as e:
        print("Confidence error:", e)
        return 75.0

# =========================
# PREPROCESS INPUT PARU
# =========================
def preprocess_input_paru(data):
    gender = 1 if data['gender'] == 'pria' else 0
    yellow_fingers = 1 if data['yellow_fingers'] == 'yes' else 0
    chronic_disease = 1 if data['chronic_disease'] == 'yes' else 0
    fatigue = 1 if data['fatigue'] == 'yes' else 0
    cough = 1 if data['cough'] == 'yes' else 0
    shortness = 1 if data['shortness'] == 'yes' else 0
    chest_pain = 1 if data['chest_pain'] == 'yes' else 0

    features = np.array([[
        gender,
        data['age'],
        data['smoking'],
        yellow_fingers,
        chronic_disease,
        fatigue,
        cough,
        shortness,
        chest_pain
    ]])
    return features

# =========================
# UI - SIDEBAR NAVIGASI
# =========================
st.sidebar.title("🏥 Menu Prediksi")
menu = st.sidebar.radio(
    "Pilih Prediksi:",
    ["📊 Prediksi Stunting", "🫁 Prediksi Kanker Paru"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Catatan:**\n\n"
    "Hasil prediksi ini hanya bersifat estimasi.\n\n"
    "Konsultasikan dengan tenaga medis profesional untuk diagnosis akurat."
)

# =========================
# HALAMAN PREDIKSI STUNTING
# =========================
if menu == "📊 Prediksi Stunting":
    st.title("📊 Prediksi Stunting pada Balita")
    st.markdown("Masukkan data balita untuk memprediksi status pertumbuhan.")

    col1, col2, col3 = st.columns(3)

    with col1:
        umur = st.number_input(
            "Umur (bulan)",
            min_value=0,
            max_value=60,
            value=24,
            step=1
        )

    with col2:
        jk = st.selectbox(
            "Jenis Kelamin",
            options=[0, 1],
            format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki"
        )

    with col3:
        tinggi = st.number_input(
            "Tinggi Badan (cm)",
            min_value=30.0,
            max_value=120.0,
            value=80.0,
            step=0.1
        )

    if st.button("🔮 Prediksi Stunting", type="primary", use_container_width=True):
        with st.spinner("Memproses prediksi..."):
            try:
                data = np.array([[umur, jk, tinggi]])
                data_scaled = scaler_stunting.transform(data)
                hasil = model_stunting.predict(data_scaled)
                hasil_text = label_map_stunting.get(hasil[0], "Tidak diketahui")
                recommendation = get_stunting_recommendation(hasil_text)

                st.markdown("---")
                st.subheader("📋 Hasil Prediksi")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Status", hasil_text)

                with col2:
                    st.metric("Tingkat Risiko", recommendation["risk"])

                with col3:
                    st.metric("Umur", f"{umur} bulan")

                st.info(f"💡 **Rekomendasi:** {recommendation['message']}")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")

# =========================
# HALAMAN PREDIKSI KANKER PARU
# =========================
elif menu == "🫁 Prediksi Kanker Paru":
    st.title("🫁 Prediksi Kanker Paru-Paru")
    st.markdown("Masukkan data pasien untuk deteksi dini kanker paru-paru.")

    with st.form("form_paru"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Jenis Kelamin",
                options=["pria", "wanita"],
                format_func=lambda x: "Pria" if x == "pria" else "Wanita"
            )

            age = st.number_input(
                "Usia (tahun)",
                min_value=18,
                max_value=100,
                value=45,
                step=1
            )

            smoking = st.number_input(
                "Jumlah Rokok per Hari",
                min_value=0,
                max_value=80,
                value=10,
                step=1
            )

            yellow_fingers = st.selectbox(
                "Jari Kuning",
                options=["no", "yes"],
                format_func=lambda x: "Tidak" if x == "no" else "Ya"
            )

            chronic_disease = st.selectbox(
                "Penyakit Kronis",
                options=["no", "yes"],
                format_func=lambda x: "Tidak" if x == "no" else "Ya"
            )

        with col2:
            fatigue = st.selectbox(
                "Kelelahan",
                options=["no", "yes"],
                format_func=lambda x: "Tidak" if x == "no" else "Ya"
            )

            cough = st.selectbox(
                "Batuk",
                options=["no", "yes"],
                format_func=lambda x: "Tidak" if x == "no" else "Ya"
            )

            shortness = st.selectbox(
                "Sesak Napas",
                options=["no", "yes"],
                format_func=lambda x: "Tidak" if x == "no" else "Ya"
            )

            chest_pain = st.selectbox(
                "Nyeri Dada",
                options=["no", "yes"],
                format_func=lambda x: "Tidak" if x == "no" else "Ya"
            )

        submitted = st.form_submit_button("🔮 Prediksi Kanker Paru", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Memproses prediksi..."):
            try:
                input_data = {
                    'gender': gender,
                    'age': age,
                    'smoking': smoking,
                    'yellow_fingers': yellow_fingers,
                    'chronic_disease': chronic_disease,
                    'fatigue': fatigue,
                    'cough': cough,
                    'shortness': shortness,
                    'chest_pain': chest_pain
                }

                features = preprocess_input_paru(input_data)
                X_scaled = scaler_paru.transform(features)
                prediction = model_paru.predict(X_scaled)[0]
                confidence = calculate_confidence_paru(model_paru, X_scaled)

                result_text = label_map_paru.get(prediction, "Tidak diketahui")
                risk_level = get_paru_risk_level(prediction, confidence)
                recommendation = get_paru_recommendation(prediction, confidence)

                st.markdown("---")
                st.subheader("📋 Hasil Prediksi")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Diagnosis", result_text)

                with col2:
                    st.metric("Tingkat Risiko", risk_level)

                with col3:
                    st.metric("Keyakinan", f"{confidence}%")

                st.info(f"💡 **Rekomendasi:** {recommendation}")

                st.markdown("---")
                st.subheader("📊 Detail Data Pasien")
                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.write(f"**Jenis Kelamin:** {'Pria' if gender == 'pria' else 'Wanita'}")
                    st.write(f"**Usia:** {age} tahun")
                    st.write(f"**Rokok/hari:** {smoking} batang")
                    st.write(f"**Jari Kuning:** {'Ya' if yellow_fingers == 'yes' else 'Tidak'}")
                    st.write(f"**Penyakit Kronis:** {'Ya' if chronic_disease == 'yes' else 'Tidak'}")

                with detail_col2:
                    st.write(f"**Kelelahan:** {'Ya' if fatigue == 'yes' else 'Tidak'}")
                    st.write(f"**Batuk:** {'Ya' if cough == 'yes' else 'Tidak'}")
                    st.write(f"**Sesak Napas:** {'Ya' if shortness == 'yes' else 'Tidak'}")
                    st.write(f"**Nyeri Dada:** {'Ya' if chest_pain == 'yes' else 'Tidak'}")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("⚠️ Prediksi ini hanya bersifat edukasi. Selalu konsultasikan dengan tenaga medis profesional.")