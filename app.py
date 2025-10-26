# ======================================================
# 🌾 SISTEM REKOMENDASI TANAMAN BERDASARKAN LINGKUNGAN
# ======================================================

# -----------------------------
# 🧩 Import Library
# -----------------------------
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import altair as alt

# ----------------
# ⚙️ Page Config 
# ----------------
st.set_page_config(page_title="Sistem Rekomendasi Tanaman", layout="wide")

# -----------------------------
# 🎨 Styling (Tampilan Modern)
# -----------------------------
st.markdown("""
<style>
    .main {
        background-color: #f6fff5;
    }
    h1, h2, h3 {
        color: #2e7d32;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #43a047;
        color: #f1f1f1;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 🧭 Judul Halaman
# -----------------------------
st.title("🌾 Sistem Rekomendasi Tanaman")
st.caption("Gunakan kondisi tanah dan cuaca untuk mengetahui tanaman terbaik yang bisa ditanam.")

# -----------------------------
# 📚 Sidebar Informasi
# -----------------------------
with st.sidebar:
    st.markdown("""
    ## 🌿 Tentang Aplikasi
    ---
    **Sistem Rekomendasi Tanaman** ini membantu menentukan **jenis tanaman paling sesuai**
    berdasarkan kondisi lingkungan seperti:

    🌱 **Parameter Input:**
    - Nitrogen (N)
    - Fosfor (P)
    - Kalium (K)
    - Suhu (°C)
    - Kelembaban (%)
    - pH Tanah
    - Curah Hujan (mm)

    ⚙️ **Model yang Digunakan:**
    *Ensemble Deep Learning* (gabungan **ANN + DNN + CNN**)

    📊 **Sumber Dataset:**
    *[Crop Recommendation Dataset – Kaggle](https://www.kaggle.com/datasets/varshitanalluri/crop-recommendation-dataset)*

    ---
    💡 **Tujuan:**
    Membantu petani dan peneliti menentukan tanaman optimal  
    untuk meningkatkan produktivitas pertanian 🌾
    """)

# -----------------------------
# 🧩 Load Model, Scaler, dan Encoder
# -----------------------------
MODEL_PATH = "ensemble_model.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

@st.cache_resource
def load_tools():
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

ensemble_model, scaler, label_encoder = load_tools()

# -----------------------------
# 📘 Panduan Pengisian
# -----------------------------
st.markdown("### 📌 Panduan Pengisian Parameter")

col1, col2 = st.columns([2, 1])  

with col1:
    st.markdown("""
    | Parameter         | Rentang Disarankan |
    |-------------------|--------------------|
    | Nitrogen (N)      | 0 – 129            |
    | Fosfor (P)        | 6 – 143            |
    | Kalium (K)        | 8 – 204            |
    | Suhu (°C)         | 12 – 41            |
    | Kelembaban (%)    | 15 – 97            |
    | pH Tanah          | 4.6 – 8.7          |
    | Curah Hujan (mm)  | 22 – 268           |
    """)

with col2:
    st.image("tanaman.png", caption="Contoh Tanaman Rekomendasi", use_container_width=True)

# -----------------------------
# 🌿 Daftar Jenis Tanaman
# -----------------------------
st.markdown("### 🌱 Daftar Jenis Tanaman yang Dapat Direkomendasikan")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    | No | Nama Lokal (Indonesia) | Nama Asli (English) |
    |----|-------------------------|---------------------|
    | 1  | Padi                   | Rice                |
    | 2  | Jagung                 | Maize               |
    | 3  | Kacang Arab            | Chickpea            |
    | 4  | Kacang Merah           | Kidney Beans        |
    | 5  | Kacang Gude            | Pigeon Peas         |
    | 6  | Kacang Ngengat         | Moth Beans          |
    | 7  | Kacang Hijau           | Mung Bean           |
    | 8  | Kacang Hitam           | Black Gram          |
    | 9  | Lentil                 | Lentil              |
    | 10 | Delima                 | Pomegranate         |
    | 11 | Pisang                 | Banana              |
    """)

with col2:
    st.markdown("""
    | No | Nama Lokal (Indonesia) | Nama Asli (English) |
    |----|-------------------------|---------------------|
    | 12 | Mangga                 | Mango               |
    | 13 | Anggur                 | Grapes              |
    | 14 | Semangka               | Watermelon          |
    | 15 | Melon                  | Muskmelon           |
    | 16 | Apel                   | Apple               |
    | 17 | Jeruk                  | Orange              |
    | 18 | Pepaya                 | Papaya              |
    | 19 | Kelapa                 | Coconut             |
    | 20 | Kapas                  | Cotton              |
    | 21 | Rami / Goni            | Jute                |
    | 22 | Kopi                   | Coffee              |
    """)

# -----------------------------
# 🌤️ Input Form
# -----------------------------
st.header("Masukkan Parameter Tanah dan Cuaca")

col1, col2, col3 = st.columns(3)
with col1:
    N = st.text_input("Nitrogen (N)", placeholder="Masukkan nilai N (0–129)")
    temperature = st.text_input("Suhu (°C)", placeholder="Masukkan suhu (12–41)")
    ph = st.text_input("pH Tanah", placeholder="Masukkan pH tanah (4.6–8.7)")

with col2:
    P = st.text_input("Fosfor (P)", placeholder="Masukkan nilai P (6–143)")
    humidity = st.text_input("Kelembaban (%)", placeholder="Masukkan kelembaban (15–97)")
    rainfall = st.text_input("Curah Hujan (mm)", placeholder="Masukkan curah hujan (22–268)")

with col3:
    K = st.text_input("Kalium (K)", placeholder="Masukkan nilai K (8–204)")

# -----------------------------
# 🔍 Prediksi Tanaman
# -----------------------------
if st.button("🔍 Prediksi Tanaman Terbaik"):
    try:
        input_values = [
            float(N), float(P), float(K),
            float(temperature), float(humidity),
            float(ph), float(rainfall)
        ]
        input_data = np.array([input_values])
        input_scaled = scaler.transform(input_data)

        pred = ensemble_model.predict(input_scaled, verbose=0)
        predicted_index = np.argmax(pred)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        st.success(f"🌾 **{predicted_label.capitalize()}** adalah tanaman paling cocok untuk kondisi lingkungan yang kamu masukkan!")

        # Grafik probabilitas interaktif
        st.markdown("---")
        st.subheader("📊 Probabilitas Semua Tanaman:")
        prob_dict = {label_encoder.classes_[i]: float(pred[0][i]) for i in range(len(label_encoder.classes_))}
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        prob_df = pd.DataFrame({
            "Tanaman": list(sorted_probs.keys()),
            "Probabilitas": list(sorted_probs.values())
        })

        chart = alt.Chart(prob_df).mark_bar(color="#66bb6a").encode(
            y=alt.Y("Probabilitas", axis=alt.Axis(format='%')),
            x=alt.X("Tanaman", sort=None),
            tooltip=["Tanaman", alt.Tooltip("Probabilitas", format=".2%")]
        ).properties(
            height=400
        )

        st.altair_chart(chart, use_container_width=True)


    except ValueError:
        st.error("⚠️ Pastikan semua kolom sudah diisi dengan angka yang valid.")

# -----------------------------
# 🧠 Penjelasan Model Ensemble
# -----------------------------
with st.expander("🧠 Tentang Model Ensemble"):
    st.markdown("""
    Model ini merupakan gabungan dari tiga arsitektur:
    - **ANN (Artificial Neural Network):** untuk representasi data sederhana
    - **DNN (Deep Neural Network):** menangkap pola non-linear kompleks
    - **CNN 1D (Convolutional Neural Network):** mendeteksi hubungan antar fitur lingkungan

    Output ketiganya digabung (rata-rata) untuk menghasilkan prediksi akhir yang lebih stabil dan akurat.
    """)

# -----------------------------
# 🧾 Footer
# -----------------------------
st.markdown("""
---
<p style='text-align:center; color:gray;'>
Dikembangkan oleh <b>Fina Dwi Aulia</b> | © 2025 Sistem Rekomendasi Tanaman
</p>
""", unsafe_allow_html=True)
