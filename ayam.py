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

# ==========================================
# 1. KONFIGURASI HALAMAN & API KEY
# ==========================================
st.set_page_config(
    page_title="ChikInspect - AI Diagnosis",
    page_icon="🐔",
    layout="wide"
)

# SETUP GEMINI (CHATBOT) 
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

# SETUP MODEL KLASIFIKASI KOKSIDIOSIS/SALMONELLA/NCD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chikinspect_model_cropped_final.keras")
CLASS_NAMES = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# INISIALISASI SESSION STATE 
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'current_analysis' not in st.session_state:
    st.session_state['current_analysis'] = None 
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
        st.error("❌ Gagal memuat model TensorFlow. Pastikan file .keras tersedia di direktori yang benar.")
        return None

model = load_classifier_model()
if model is None:
    st.stop()

def run_roboflow_detection(image_bytes):
    try:
        api_key = st.secrets["roboflow_api_key"]
    except KeyError:
        st.warning("⚠️ Roboflow API Key belum dikonfigurasi di secrets.toml")
        return None

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = "https://detect.roboflow.com/infer/workflows/elvin-3wtt1/find-feses-3"
    
    payload = {
        "api_key": api_key,
        "inputs": {
            "image": {
                "type": "base64",
                "value": img_b64
            }
        }
    }
    
    headers = {"Content-Type": "application/json"}

    try:
        # allow_redirects=False mencegah mutasi otomatis POST ke GET
        response = requests.post(url, json=payload, headers=headers, allow_redirects=False)
        response.raise_for_status() 
        return response.json()
        
    except requests.exceptions.RequestException as e:
        st.error("Kegagalan sinkronisasi dengan peladen Workflows.")
        if getattr(e, 'response', None) is not None:
            st.error(f"Detail Diagnostik Peladen: {e.response.text}")
        return None

def extract_predictions(resp):
    if not isinstance(resp, dict):
        return []
    if "predictions" in resp:
        return resp.get("predictions", [])
    if "outputs" in resp and isinstance(resp["outputs"], list):
        try:
            return resp["outputs"][0].get("predictions", [])
        except (IndexError, AttributeError):
            pass
    return []

def draw_bounding_boxes(image, preds):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for p in preds:
        try:
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            # Konversi koordinat tengah menjadi titik sudut (Top-Left, Bottom-Right)
            draw.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline="red", width=3)
            label = p.get("class", "feses")
            conf = p.get("confidence", 0)
            draw.text((x-w/2, y-h/2 - 10), f"{label} ({conf:.2f})", fill="red")
        except Exception: 
            continue
    return img

def predict_crop(crop_img, model):
    if model is None: 
        return "Unknown", 0.0
    # Pastikan ukuran sejalan dengan input layer model Anda
    img = crop_img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return CLASS_NAMES[class_idx], confidence

# ==========================================
# 3. FUNGSI-FUNGSI CHATBOT LLM
# ==========================================
def build_system_prompt():
    base = SYSTEM_PROMPT
    if st.session_state.current_analysis and "error" not in st.session_state.current_analysis:
        base += f"\n\n[KONTEKS MEDIS SAAT INI]\n{st.session_state.current_analysis['context_str']}"
    return base

def ask_gemini(messages):
    contents = []
    for m in messages:
        if m["role"] != "system": 
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
    if GEMINI_API_KEY_1: api_keys_to_try.append(GEMINI_API_KEY_1)
    if GEMINI_API_KEY_2: api_keys_to_try.append(GEMINI_API_KEY_2)

    if not api_keys_to_try:
        return "Galat: API Key Gemini tidak terdeteksi pada environment."

    last_error = None
    for api_key in api_keys_to_try:
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
    
    return f"Maaf, gagal memproses respons AI: {str(last_error)}"

# ==========================================
# 4. ANTARMUKA SIDEBAR
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.title("📂 Riwayat Diagnosa")
        st.info("Data hanya tersimpan di memori sesi aktif.")
        
        if len(st.session_state['history']) > 0:
            df_hist = pd.DataFrame(st.session_state['history'])
            st.dataframe(df_hist[['Waktu', 'Diagnosa', 'Skor']], hide_index=True)
            csv = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh CSV", csv, "riwayat_inspeksi.csv", "text/csv")
            if st.button("🗑️ Kosongkan Riwayat"):
                st.session_state['history'] = []
                st.rerun()
        else:
            st.caption("Belum ada data diagnosa.")

# ==========================================
# 5. HALAMAN UTAMA (MAIN APP)
# ==========================================
def main():
    render_sidebar()
    
    st.title("🐔 ChikInspect AI - Deteksi & Konsultasi")
    st.markdown("Unggah foto visual feses unggas untuk pemindaian penyakit. Konsultasikan hasilnya bersama Dokter AI di panel bawah.")
    st.divider()

    # BAGIAN 1: UPLOAD DAN DETEKSI 
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    # Reset manajemen state jika file berubah
    if uploaded_file:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.messages = [] 
            st.session_state.current_analysis = None 
            st.session_state.last_uploaded_file = uploaded_file.name

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Citra Mentah", use_column_width=True)

        analyze_clicked = st.button("🔍 Deteksi Objek Gejala", type="primary", use_container_width=True)

        if analyze_clicked:
            with st.spinner('Memanggil infrastruktur jaringan visi...'):
                raw_resp = run_roboflow_detection(image_bytes)
                
                # IMPLEMENTASI FAIL-FAST: Cegah NameError
                if raw_resp is None:
                    st.warning("⚠️ Proses dihentikan otomatis akibat galat koneksi ke server deteksi.")
                    st.stop()
                    
                predictions = extract_predictions(raw_resp)
                bbox_image = draw_bounding_boxes(original_image, predictions)
                
                # Saringan objek berdasarkan rasio keyakinan (confidence ratio)
                valid_predictions = [p for p in predictions if p.get('confidence', 0) >= 0.5]
                results = []
                
                for pred in valid_predictions:
                    x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                    
                    # Validasi batas dimensi matriks (Boundary Clipping)
                    left = max(0, x - w/2)
                    top = max(0, y - h/2)
                    right = min(original_image.width, x + w/2)
                    bottom = min(original_image.height, y + h/2)
                    
                    crop_img = original_image.crop((left, top, right, bottom))
                    label, disease_conf = predict_crop(crop_img, model)
                    
                    # Kalkulasi probabilitas gabungan (Joint Probability)
                    final_score = pred['confidence'] * disease_conf
                    
                    results.append({
                        "disease": label,
                        "final_score": final_score,
                        "roboflow": pred['confidence'],
                        "classifier": disease_conf
                    })

                if results:
                    best_pred = max(results, key=lambda x: x['final_score'])
                    
                    context_str = f"""
                    Hasil komputasi asimetris AI terbaru:
                    - Patologi Tertinggi: {best_pred['disease']}
                    - Skor Joint Probability: {best_pred['final_score']:.3f}
                    - Waktu Perekaman: {datetime.now().strftime('%H:%M:%S')}
                    """
                    
                    st.session_state.current_analysis = {
                        "bbox_image": bbox_image,
                        "best_result": best_pred,
                        "context_str": context_str
                    }
                    
                    st.session_state['history'].append({
                        "Waktu": datetime.now().strftime("%H:%M:%S"),
                        "Nama File": uploaded_file.name,
                        "Diagnosa": best_pred['disease'],
                        "Skor": round(best_pred['final_score'], 3)
                    })
                    
                    st.session_state.messages = [{
                        "role": "assistant", 
                        "content": f"**Pemindaian Klinis Selesai.**\nBerdasarkan ekstraksi fitur, saya mendeteksi indikasi **{best_pred['disease']}** (Skor Keyakinan: {best_pred['final_score']:.2f}). Apakah ada variabel tambahan terkait kondisi flok unggas yang ingin Anda diskusikan?"
                    }]
                else:
                    st.session_state.current_analysis = {"error": "Tidak ada objek valid."}
                    st.warning("Model deteksi objek tidak menemukan region feses dengan tingkat keyakinan di atas batas ambang (threshold 0.5).")

        # TAMPILAN PERSISTEN
        if st.session_state.current_analysis and "error" not in st.session_state.current_analysis:
            data = st.session_state.current_analysis
            
            with col2:
                st.image(data["bbox_image"], caption="Visualisasi Bounding Box", use_column_width=True)
            
            st.success(f"### ✅ Klasifikasi Final: {data['best_result']['disease']}")
            st.progress(float(data['best_result']['final_score']))

    # BAGIAN 2: ANTARMUKA LLM 
    st.divider()
    st.subheader("💬 Agen Diskusi Medis")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.current_analysis is None:
        st.caption("ℹ️ Sistem agen ini membutuhkan pemuatan konteks visual dari pemicu di atas untuk memberikan diagnosis akurat.")

    if prompt := st.chat_input("Berikan parameter tambahan (umur unggas, suhu, jenis pakan) untuk akurasi rekomendasi..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Mensintesis informasi protokol medis..."):
                response_text = ask_gemini(st.session_state.messages)
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()
