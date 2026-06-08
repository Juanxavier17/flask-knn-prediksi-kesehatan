from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# =========================
# MODEL STUNTING
# =========================
model_stunting = joblib.load("knn_model.pkl")
scaler_stunting = joblib.load("scaler.pkl")

label_map_stunting = {
    0: "Sangat Pendek",
    1: "Pendek",
    2: "Normal",
    3: "Tinggi"
}

# =========================
# MODEL KANKER PARU
# =========================
model_paru = joblib.load("model_paru_pas.pkl")
scaler_paru = joblib.load("scaler_paru.pkl")

label_map_paru = {
    0: "Tidak Terdeteksi Kanker",
    1: "Terdeteksi Kanker Paru-Paru"
}

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
# CONFIDENCE
# =========================
def calculate_confidence_paru(model, X_scaled):

    try:

        if hasattr(model, 'kneighbors'):

            k = model.n_neighbors

            distances, indices = model.kneighbors(X_scaled)

            neighbor_labels = model._y[indices[0]]

            prediction = model.predict(X_scaled)[0]

            same_label_count = np.sum(
                neighbor_labels == prediction
            )

            confidence = (
                same_label_count / k
            ) * 100

            return round(confidence, 1)

        return 75.0

    except Exception as e:

        print("Confidence error:", e)

        return 75.0

# =========================
# REKOMENDASI PARU
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

# =========================
# RISK LEVEL PARU
# =========================
def get_paru_risk_level(prediction, confidence):

    if prediction == 1:
        return "high" if confidence >= 60 else "moderate"

    return "low" if confidence >= 60 else "moderate"

# =========================
# REKOMENDASI STUNTING
# =========================
def get_stunting_recommendation(status):

    if status == "Sangat Pendek":

        return {
            "risk": "high",
            "message": "🔴 Segera konsultasi ke dokter anak atau ahli gizi untuk pemeriksaan pertumbuhan dan pola makan balita."
        }

    elif status == "Pendek":

        return {
            "risk": "moderate",
            "message": "🟠 Perlu perhatian terhadap asupan gizi dan pemeriksaan rutin pertumbuhan anak."
        }

    elif status == "Normal":

        return {
            "risk": "low",
            "message": "✅ Pertumbuhan balita normal. Tetap jaga pola makan sehat dan rutin cek pertumbuhan."
        }

    elif status == "Tinggi":

        return {
            "risk": "low",
            "message": "✅ Tinggi badan di atas rata-rata. Tetap perhatikan keseimbangan nutrisi anak."
        }

    return {
        "risk": "moderate",
        "message": "⚠️ Data tidak dikenali."
    }

# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():

    return render_template(
        "index.html",
        stunting_result=None,
        paru_result=None,
        paru_error=None
    )
# =========================
# PREDIKSI STUNTING
# =========================
@app.route("/predict", methods=["POST"])
def predict_stunting():

    try:

        umur = float(request.form["umur"])
        jk = int(request.form["jk"])
        tinggi = float(request.form["tinggi"])

        data = np.array([[umur, jk, tinggi]])

        data_scaled = scaler_stunting.transform(data)

        hasil = model_stunting.predict(data_scaled)

        hasil_text = label_map_stunting.get(
            hasil[0],
            "Tidak diketahui"
        )

        recommendation = get_stunting_recommendation(
            hasil_text
        )

        stunting_result = {
            "prediction": hasil_text,
            "risk": recommendation["risk"],
            "message": recommendation["message"]
        }

        return render_template(
            "index.html",
            stunting_result=stunting_result,
            paru_result=None,
            paru_error=None
        )

    except Exception as e:

        return render_template(
            "index.html",
            stunting_result={
                "prediction": "Error",
                "risk": "moderate",
                "message": f"Terjadi kesalahan: {str(e)}"
            },
            paru_result=None,
            paru_error=None
        )


# =========================
# PREDIKSI PARU
# =========================
@app.route("/predict_paru", methods=["POST"])
def predict_paru():

    try:

        input_data = {

            'gender': request.form.get("gender"),

            'age': float(request.form.get("age")),

            'smoking': float(request.form.get("smoking")),

            'yellow_fingers': request.form.get("yellow_fingers"),

            'chronic_disease': request.form.get("chronic_disease"),

            'fatigue': request.form.get("fatigue"),

            'cough': request.form.get("cough"),

            'shortness': request.form.get("shortness"),

            'chest_pain': request.form.get("chest_pain")
        }

        if not input_data['gender']:

            return render_template(
                "index.html",
                paru_error="Gender wajib dipilih",
                stunting_result=None,
                paru_result=None
            )

        if input_data['age'] < 18 or input_data['age'] > 100:

            return render_template(
                "index.html",
                paru_error="Usia harus 18-100",
                stunting_result=None,
                paru_result=None
            )

        if input_data['smoking'] < 0 or input_data['smoking'] > 80:

            return render_template(
                "index.html",
                paru_error="Rokok harus 0-80 batang",
                stunting_result=None,
                paru_result=None
            )

        # PREPROCESS
        features = preprocess_input_paru(input_data)

        # SCALE
        X_scaled = scaler_paru.transform(features)

        # PREDIKSI
        prediction = model_paru.predict(X_scaled)[0]

        # CONFIDENCE
        confidence = calculate_confidence_paru(
            model_paru,
            X_scaled
        )

        result_text = label_map_paru.get(
            prediction,
            "Tidak diketahui"
        )

        risk_level = get_paru_risk_level(
            prediction,
            confidence
        )

        recommendation = get_paru_recommendation(
            prediction,
            confidence
        )

        paru_result = {

            'prediction': int(prediction),

            'prediction_label': result_text,

            'confidence': confidence,

            'risk_level': risk_level,

            'recommendation': recommendation,

            'gender': (
                'Pria'
                if input_data['gender'] == 'pria'
                else 'Wanita'
            ),

            'age': int(input_data['age']),

            'smoking': int(input_data['smoking']),

            'yellow_fingers': (
                'Ya'
                if input_data['yellow_fingers'] == 'yes'
                else 'Tidak'
            ),

            'chronic_disease': (
                'Ya'
                if input_data['chronic_disease'] == 'yes'
                else 'Tidak'
            ),

            'fatigue': (
                'Ya'
                if input_data['fatigue'] == 'yes'
                else 'Tidak'
            ),

            'cough': (
                'Ya'
                if input_data['cough'] == 'yes'
                else 'Tidak'
            ),

            'shortness': (
                'Ya'
                if input_data['shortness'] == 'yes'
                else 'Tidak'
            ),

            'chest_pain': (
                'Ya'
                if input_data['chest_pain'] == 'yes'
                else 'Tidak'
            )
        }

        return render_template(
            "index.html",
            paru_result=paru_result,
            stunting_result=None,
            paru_error=None
        )

    except Exception as e:

        print("Prediction error:", e)

        return render_template(
            "index.html",
            paru_error=f"Terjadi kesalahan: {str(e)}",
            stunting_result=None,
            paru_result=None
        )


# =========================
# RUN APP
# =========================
if __name__ == "__main__":

    print(app.url_map)

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )