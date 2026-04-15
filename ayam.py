import streamlit as st
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import requests
import io
from datetime import datetime
import base64
import json

# Import Library Gemini
from google import genai
from google.genai.errors import APIError

# ==========================================
# 1. KONFIGURASI HALAMAN & API KEY
# ==========================================
st.set_page_config(
    page_title="ChikInspect - AI Diagnosis",
    page_icon="🐔",
    layout="wide"
)

# --- SETUP GEMINI (CHATBOT) ---
GEMINI_API_KEY_1 = os.environ.get('GEMINI_API_KEY_1')
if not GEMINI_API_KEY_1 and "GEMINI_API_KEY_1" in st.secrets:
    GEMINI_API_KEY_1 = st.secrets["GEMINI_API_KEY_1"]

GEMINI_API_KEY_2 = os.environ.get('GEMINI_API_KEY_2')
if not GEMINI_API_KEY_2 and "GEMINI_API_KEY_2" in st.secrets:
    GEMINI_API_KEY_2 = st.secrets["GEMINI_API_KEY_2"]

if not GEMINI_API_KEY_1:
    GEMINI_API_KEY_1 = os.environ.get('GEMINI_API_KEY')
    if not GEMINI_API_KEY_1 and "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY_1 = st.secrets["GEMINI_API_KEY"]

# Instruksi Dokter AI
SYSTEM_PROMPT = (
    "Anda adalah Dokter AI ahli kesehatan unggas. "
    "Analisis gejala berdasarkan konteks yang diberikan, berikan saran obat (misal: Amprolium untuk Koksidiosis), dan pencegahan. "
    "Jawab singkat, padat, dan profesional dalam Bahasa Indonesia."
)

# --- SETUP MODEL KLASIFIKASI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chikinspect_model_cropped_final.keras")
CLASS_NAMES = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# --- INISIALISASI SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'current_analysis' not in st.session_state:
    st.session_state['current_analysis'] = None # Menyimpan hasil deteksi aktif
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# ==========================================
# 2. FUNGSI-FUNGSI LOGIKA (ML & ROBOFLOW)
# ==========================================
@st.cache_resource
def load_classifier_model():
    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
    except Exception as e:
        st.error("❌ Gagal load model")
        st.error(e)
        return None

model = load_classifier_model()
if model is None:
    st.stop()

def run_roboflow_detection(image_bytes):
    try:
        api_key = st.secrets["roboflow_api_key"]
    except:
        st.warning("⚠️ Roboflow API Key belum diset di secrets.toml")
        return None

    # Encode gambar ke base64 string
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # URL Endpoint API langsung ke Roboflow Workflows
    url = f"https://serverless.roboflow.com/v1/workspaces/elvin-3wtt1/workflows/find-feses-3?api_key={api_key}"
    
    # Struktur JSON sesuai protokol Roboflow API
    payload = {
        "inputs": {
            "image": {
                "type": "base64",
                "value": img_b64
            }
        }
    }
    
    headers = {"Content-Type": "application/json"}

    try:
        # Eksekusi HTTP POST
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status() # Lemparkan error jika status bukan 200 OK
        
        resp = response.json()
        
        # Penyesuaian format output agar kompatibel dengan sisa kode Anda
        if isinstance(resp, list): 
            resp = resp[0]
        return resp
        
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal menghubungi server Roboflow: {e}")
        return None

# ==========================================
# 3. FUNGSI-FUNGSI CHATBOT
# ==========================================
def build_system_prompt():
    base = SYSTEM_PROMPT
    # Ambil konteks dari hasil analisis terakhir yang tersimpan
    if st.session_state.current_analysis and "error" not in st.session_state.current_analysis:
        base += f"\n\n[KONTEKS MEDIS SAAT INI]\n{st.session_state.current_analysis['context_str']}"
    return base

def ask_gemini(messages):
    contents = []
    for m in messages:
        if m["role"] != "system": # Pastikan system prompt tidak masuk sini
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": m["content"]}]
            })

    system_prompt = build_system_prompt()
    config = {
        "system_instruction": {"parts": [{"text": system_prompt}]}
    }

    api_keys_to_try = []
    if GEMINI_API_KEY_1: api_keys_to_try.append(("GEMINI_API_KEY_1", GEMINI_API_KEY_1))
    if GEMINI_API_KEY_2: api_keys_to_try.append(("GEMINI_API_KEY_2", GEMINI_API_KEY_2))

    if not api_keys_to_try:
        return "Maaf, API Key Gemini tidak tersedia."

    last_error = None
    for api_key_name, api_key in api_keys_to_try:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=config
            )
            return response.text
        except Exception as e:
            last_error = e
            continue
    
    return f"Maaf, terjadi kesalahan atau kuota habis: {str(last_error)}"

# ==========================================
# 4. SIDEBAR
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.title("📂 Riwayat Diagnosa")
        st.info("Data hilang jika di-refresh.")
        
        if len(st.session_state['history']) > 0:
            df_hist = pd.DataFrame(st.session_state['history'])
            st.dataframe(df_hist[['Waktu', 'Diagnosa', 'Skor']], hide_index=True)
            csv = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "riwayat.csv", "text/csv")
            if st.button("🗑️ Hapus Riwayat"):
                st.session_state['history'] = []
                st.rerun()
        else:
            st.caption("Belum ada data.")

# ==========================================
# 5. HALAMAN UTAMA (MAIN APP)
# ==========================================
def main():
    render_sidebar()
    
    st.title("🐔 ChikInspect AI - Deteksi & Konsultasi")
    st.markdown("Upload foto feses ayam untuk deteksi penyakit, lalu konsultasikan hasilnya dengan Dokter AI di bawah.")
    st.divider()

    # --- BAGIAN 1: UPLOAD DAN DETEKSI ---
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    # Reset state jika user ganti file
    if uploaded_file:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.messages = [] # Reset chat
            st.session_state.current_analysis = None # Reset hasil diagnosa
            st.session_state.last_uploaded_file = uploaded_file.name

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Gambar Asli", use_column_width=True)

        # Tombol Deteksi
        analyze_clicked = st.button("🔍 Deteksi Penyakit", type="primary", use_container_width=True)

        # Logika Deteksi (Hanya jalan saat tombol diklik)
        if analyze_clicked:
            with st.spinner('Sedang memindai objek feses & menganalisis...'):
                raw_resp = run_roboflow_detection(image_bytes)
                predictions = extract_predictions(raw_resp)
                
                # Visualisasi Bounding Box
                bbox_image = draw_bounding_boxes(original_image, predictions)
                
                # Logic Crop & Predict Keras (Safety Mode)
                valid_predictions = [p for p in predictions if p['confidence'] >= 0.5]
                results = []
                
                for i, pred in enumerate(valid_predictions):
                    x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                    
                    # Safety Crop: Pastikan koordinat tidak minus atau melebihi gambar
                    left = max(0, x - w/2)
                    top = max(0, y - h/2)
                    right = min(original_image.width, x + w/2)
                    bottom = min(original_image.height, y + h/2)
                    
                    crop_img = original_image.crop((left, top, right, bottom))
                    label, disease_conf = predict_crop(crop_img, model)
                    final_score = pred['confidence'] * disease_conf
                    
                    results.append({
                        "disease": label,
                        "final_score": final_score,
                        "roboflow": pred['confidence'],
                        "classifier": disease_conf
                    })

                # Simpan hasil ke session_state agar persisten (tidak hilang saat chat)
                if results:
                    best_pred = max(results, key=lambda x: x['final_score'])
                    
                    context_str = f"""
                    Hasil analisis AI terbaru:
                    - Penyakit terdeteksi: {best_pred['disease']}
                    - Tingkat Keparahan/Skor: {best_pred['final_score']:.3f}
                    - Waktu: {datetime.now().strftime('%H:%M:%S')}
                    """
                    
                    st.session_state.current_analysis = {
                        "bbox_image": bbox_image,
                        "best_result": best_pred,
                        "context_str": context_str
                    }
                    
                    # Simpan ke history sidebar
                    st.session_state['history'].append({
                        "Waktu": datetime.now().strftime("%H:%M:%S"),
                        "Nama File": uploaded_file.name,
                        "Diagnosa": best_pred['disease'],
                        "Skor": round(best_pred['final_score'], 3)
                    })
                    
                    # Tambahkan pesan pembuka chat otomatis
                    st.session_state.messages = [{
                        "role": "assistant", 
                        "content": f"**Analisis Selesai.**\nSaya mendeteksi indikasi **{best_pred['disease']}** dengan skor keyakinan {best_pred['final_score']:.2f}. Ada yang ingin Anda tanyakan mengenai penanganan penyakit ini?"
                    }]
                else:
                    st.session_state.current_analysis = {"error": "Tidak ada objek terdeteksi."}
                    st.warning("Tidak ada objek feses yang terdeteksi dengan jelas.")

        # --- TAMPILKAN HASIL (PERSISTENT) ---
        # Bagian ini mengambil data dari memori, jadi tetap muncul meski halaman refresh saat chat
        if st.session_state.current_analysis and "error" not in st.session_state.current_analysis:
            data = st.session_state.current_analysis
            
            with col2:
                st.image(data["bbox_image"], caption="Hasil Deteksi", use_column_width=True)
            
            st.success(f"### ✅ Diagnosa Utama: {data['best_result']['disease']}")
            st.progress(float(data['best_result']['final_score']))

    # --- BAGIAN 2: CHATBOT (DI BAWAH HASIL) ---
    st.divider()
    st.subheader("💬 Konsultasi Dokter AI")

    # Tampilkan History Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input User
    if st.session_state.current_analysis is None:
        st.caption("ℹ️ Silakan lakukan deteksi gambar di atas untuk mengaktifkan konteks medis Dokter AI.")

    if prompt := st.chat_input("Tanya tentang pengobatan, dosis, atau pencegahan..."):
        # 1. Tampilkan pesan user langsung
        st.chat_message("user").markdown(prompt)
        # 2. Simpan ke history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 3. Proses AI dengan indikator spinner (Visual saja, tidak disimpan ke list pesan)
        with st.chat_message("assistant"):
            with st.spinner("Dokter AI sedang berpikir..."):
                response_text = ask_gemini(st.session_state.messages)
                st.markdown(response_text)
        
        # 4. Simpan jawaban AI ke history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()
