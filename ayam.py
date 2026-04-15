import streamlit as st
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient
import io
from datetime import datetime

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

# --- SETUP GEMINI (CHATBOT) DENGAN CACHE ---
@st.cache_resource
def get_gemini_client(api_key):
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Gagal koneksi Gemini: {e}")
        return None

# Konfigurasi API Keys
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or st.secrets.get("GEMINI_API_KEY")
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY') or st.secrets.get("roboflow_api_key")

client = get_gemini_client(GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "Anda adalah Dokter AI ahli kesehatan unggas. "
    "Analisis gejala, berikan saran obat (misal: Amprolium untuk Koksidiosis), dan pencegahan. "
    "Jawab singkat, padat, dan profesional dalam Bahasa Indonesia."
)

# --- SETUP MODEL KLASIFIKASI ---
MODEL_PATH = 'chickinspect_model_cropped_final.keras' 
CLASS_NAMES = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# Inisialisasi History
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ==========================================
# 2. FUNGSI-FUNGSI LOGIKA (ML & ROBOFLOW)
# ==========================================
@st.cache_resource
def load_classifier_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model {MODEL_PATH} tidak ditemukan!")
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Gagal memuat model Keras: {e}")
        return None

model = load_classifier_model()

def run_roboflow_detection(pil_image):
    if not ROBOFLOW_API_key:
        st.warning("⚠️ Roboflow API Key belum dikonfigurasi.")
        return None

    client_rf = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )
    
    try:
        # Gunakan parameter 'images' dengan format list jika workflow mengharapkan batch
        # atau tetap dict jika itu single input.
        resp = client_rf.run_workflow(
            workspace_name="elvin-3wtt1",
            workflow_id="find-feses-3",
            images={"image": pil_image}
        )
        
        # --- LOGIKA DEFENSIF UNTUK MENGATASI ERROR 'LIST' ---
        # Jika resp adalah list, ambil elemen pertama
        if isinstance(resp, list):
            if len(resp) > 0:
                resp = resp[0]
            else:
                return None
        
        # Jika setelah diekstrak masih bukan dict, kita tidak bisa lanjut
        if not isinstance(resp, dict):
            st.error(f"Format respons tidak dikenal: {type(resp)}")
            return None
            
        return resp
        
    except Exception as e:
        # Jika error 'items' muncul di dalam library, kita tangkap di sini
        if "'list' object has no attribute 'items'" in str(e):
            st.error("SDK Roboflow mengalami konflik tipe data. Coba update library: pip install -U inference-sdk")
        else:
            st.error(f"Error Roboflow Workflow: {e}")
        return None

def extract_predictions(resp):
    if not resp: return []
    # Jalur 1: Hasil dari Workflow API biasanya ada di kunci 'predictions'
    if "predictions" in resp:
        content = resp["predictions"]
        if isinstance(content, dict) and "predictions" in content:
            return content["predictions"]
        if isinstance(content, list):
            return content
    return []

def draw_bounding_boxes(image, preds):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for p in preds:
        try:
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            # Konversi koordinat tengah ke pojok untuk PIL
            left, top = x - w/2, y - h/2
            right, bottom = x + w/2, y + h/2
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            label = p.get("class", "obj")
            conf = p.get("confidence", 0)
            draw.text((left, top - 10), f"{label} ({conf:.2f})", fill="red")
        except: continue
    return img

def predict_crop(crop_img, model):
    if model is None: return "Unknown", 0.0
    # Preprocessing sesuai input model (asumsi 224x224 RGB)
    img = crop_img.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0  # Normalisasi jika diperlukan
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return CLASS_NAMES[class_idx], confidence

# ==========================================
# 3. UI UTAMA STREAMLIT
# ==========================================

with st.sidebar:
    st.title("📂 Riwayat Diagnosa")
    if st.session_state['history']:
        df_hist = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_hist[['Waktu', 'Diagnosa', 'Skor']], hide_index=True)
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "riwayat_chikinspect.csv", "text/csv")
        if st.button("🗑️ Hapus Semua Riwayat"):
            st.session_state['history'] = []
            st.rerun()
    else:
        st.info("Belum ada diagnosa tersimpan.")

st.title("🐔 ChikInspect AI")
st.write("Analisis kesehatan feses ayam berbasis Computer Vision dan LLM.")

uploaded_file = st.file_uploader("Upload Foto Feses Ayam", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Simpan di memory agar bisa digunakan berkali-kali tanpa upload ulang
    image_bytes = uploaded_file.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Gambar Input", use_container_width=True)

    if st.button("🔍 Jalankan Deteksi & Klasifikasi", type="primary"):
        with st.spinner('Menjalankan Object Detection (Roboflow)...'):
            # Gunakan objek PIL langsung ke fungsi yang sudah diperbaiki
            raw_resp = run_roboflow_detection(original_image)

        predictions = extract_predictions(raw_resp)
        
        if not predictions:
            st.warning("Tidak ditemukan objek feses. Coba foto yang lebih jelas atau dekat.")
        else:
            bbox_image = draw_bounding_boxes(original_image, predictions)
            with col2:
                st.image(bbox_image, caption=f"Hasil Deteksi: {len(predictions)} objek", use_container_width=True)

            st.divider()
            st.subheader("🔬 Hasil Analisis Per Objek")
            
            results = []
            valid_preds = [p for p in predictions if p.get('confidence', 0) >= 0.4]
            
            # Looping hasil deteksi untuk diklasifikasikan model Keras
            cols = st.columns(min(len(valid_preds), 4))
            for i, pred in enumerate(valid_preds):
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                # Crop area feses
                left, top = max(0, x - w/2), max(0, y - h/2)
                right, bottom = min(original_image.width, x + w/2), min(original_image.height, y + h/2)
                crop_img = original_image.crop((left, top, right, bottom))
                
                label, disease_conf = predict_crop(crop_img, model)
                final_score = pred['confidence'] * disease_conf
                
                results.append({
                    "disease": label,
                    "score": final_score
                })

                with cols[i % 4]:
                    st.image(crop_img, caption=f"{label} ({final_score:.2f})")

            # Kesimpulan Akhir
            if results:
                best_pred = max(results, key=lambda x: x['score'])
                st.success(f"### Diagnosa Utama: **{best_pred['disease']}**")
                
                st.session_state['history'].append({
                    "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Diagnosa": best_pred['disease'],
                    "Skor": round(best_pred['score'], 3)
                })

st.markdown("---")

# ==========================================
# 4. CHATBOT DOKTER AI (GEMINI)
# ==========================================
st.header("💬 Konsultasi Dokter Hewan AI")

if not GEMINI_API_KEY:
    st.info("Masukkan API Key Gemini untuk mengaktifkan fitur konsultasi.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Saya adalah Dokter AI ChikInspect. Berdasarkan hasil deteksi di atas, apa yang ingin Anda tanyakan?"}
        ]

    # Persistent Chat Session
    if "chat_session" not in st.session_state and client:
        st.session_state.chat_session = client.chats.create(
            model="gemini-2.0-flash", # Pastikan model name sesuai yang tersedia
            config={"system_instruction": SYSTEM_PROMPT}
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Tanyakan tentang gejala atau pengobatan..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = st.session_state.chat_session.send_message(prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Gemini Error: {e}")
