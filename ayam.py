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
# Fungsi ini memastikan client tidak putus saat refresh
@st.cache_resource
def get_gemini_client(api_key):
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Gagal koneksi Gemini: {e}")
        return None

# Ambil API Key
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY and "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Inisialisasi Client menggunakan Cache
client = get_gemini_client(GEMINI_API_KEY)

# Instruksi Dokter AI
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
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        return None

model = load_classifier_model()

def run_roboflow_detection(image_bytes):
    # Cek API Key Roboflow
    try:
        api_key = st.secrets["roboflow_api_key"]
    except:
        st.warning("⚠️ Roboflow API Key belum diset di secrets.toml")
        return None

    client_rf = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )
    
    import base64
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        resp = client_rf.run_workflow(
            workspace_name="elvin-3wtt1",
            workflow_id="find-feses-3",
            images={"image": img_b64}
        )
        if isinstance(resp, list): resp = resp[0]
        if isinstance(resp, str):
            import json
            resp = json.loads(resp)
        return resp
    except Exception as e:
        st.error(f"Error Roboflow: {e}")
        return None

def extract_predictions(resp):
    if resp is None: return []
    if "predictions" in resp and isinstance(resp["predictions"], dict):
        preds = resp["predictions"].get("predictions", [])
        if isinstance(preds, list): return preds
    if "predictions" in resp and isinstance(resp["predictions"], list):
        return resp["predictions"]
    return []

def draw_bounding_boxes(image, preds):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for p in preds:
        try:
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            draw.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline="red", width=3)
            label = p.get("class", "obj")
            conf = p.get("confidence", 0)
            draw.text((x-w/2, y-h/2 - 10), f"{label} ({conf:.2f})", fill="red")
        except: continue
    return img

def predict_crop(crop_img, model):
    if model is None: return "Unknown", 0.0
    img = crop_img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return CLASS_NAMES[class_idx], confidence

# ==========================================
# 3. UI UTAMA
# ==========================================

# --- SIDEBAR HISTORY ---
with st.sidebar:
    st.title("📂 Riwayat Diagnosa")
    if len(st.session_state['history']) > 0:
        df_hist = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_hist[['Waktu', 'Diagnosa', 'Skor']], hide_index=True)
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "riwayat.csv", "text/csv")
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state['history'] = []
            st.rerun()
    else:
        st.info("Belum ada data.")

# --- MAIN CONTENT ---
st.title("🐔 ChikInspect AI")
st.markdown("Upload foto feses ayam untuk mendeteksi penyakit secara otomatis.")

# UPLOAD FILE
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Gambar Asli", use_column_width=True)

    # --- BAGIAN DETEKSI ML ---
    if st.button("🔍 Deteksi Penyakit"):
        with st.spinner('Sedang memindai objek feses (Roboflow)...'):
            raw_resp = run_roboflow_detection(image_bytes)

        predictions = extract_predictions(raw_resp)
        bbox_image = draw_bounding_boxes(original_image, predictions)

        with col2:
            st.image(bbox_image, caption=f"Terdeteksi {len(predictions)} Objek", use_column_width=True)

        st.divider()
        st.subheader("🔬 Hasil Analisis Laboratorium AI")
            
        results = []
        progress_bar = st.progress(0)
        valid_predictions = [p for p in predictions if p['confidence'] >= 0.5]

        if not valid_predictions:
            st.warning("⚠️ Tidak ada objek feses yang terdeteksi dengan jelas.")
        else:
            for i, pred in enumerate(valid_predictions):
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                crop_img = original_image.crop((x-w/2, y-h/2, x+w/2, y+h/2))
                
                label, disease_conf = predict_crop(crop_img, model)
                final_score = pred['confidence'] * disease_conf
                
                results.append({
                    "id": i+1,
                    "disease": label,
                    "final_score": final_score,
                    "img": crop_img
                })
                progress_bar.progress((i + 1) / len(valid_predictions))
            
            progress_bar.empty()

            # Tampilkan Hasil Gambar Kecil
            cols = st.columns(min(len(results), 3))
            for idx, res in enumerate(results):
                with cols[idx % 3]:
                    st.image(res['img'], width=100)
                    st.caption(f"**{res['disease']}**")

            # Kesimpulan & Simpan
            if results:
                best_pred = max(results, key=lambda x: x['final_score'])
                st.success(f"### ✅ Diagnosa Utama: {best_pred['disease']}")
                
                # Simpan ke History
                st.session_state['history'].append({
                    "Waktu": datetime.now().strftime("%H:%M:%S"),
                    "Nama File": uploaded_file.name,
                    "Diagnosa": best_pred['disease'],
                    "Skor": round(best_pred['final_score'], 3)
                })
                st.toast("Data tersimpan!", icon="💾")

st.markdown("---")

# ==========================================
# 4. FITUR CHATBOT DOKTER AI (GEMINI)
# ==========================================
st.header("💬 Konsultasi dengan Dokter AI")
st.caption("Diskusikan hasil diagnosa atau tanya tips perawatan ayam...")

if not GEMINI_API_KEY:
    st.warning("⚠️ Fitur Chatbot belum aktif. Masukkan GEMINI_API_KEY di Secrets/Environment.")
else:
    # Inisialisasi History Chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "assistant", "content": "Halo! Ada yang bisa saya bantu tentang kesehatan ayam?"}
        ]

    # Inisialisasi Sesi Gemini
    if "chat_session" not in st.session_state and client:
        try:
            st.session_state.chat_session = client.chats.create(
                model="gemini-2.5-flash",
                config={"system_instruction": SYSTEM_PROMPT}
            )
        except Exception as e:
            st.error(f"Gagal init chat: {e}")

    # Tampilkan Chat
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Input User
    if prompt := st.chat_input("Ketik pertanyaan Anda..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Mengetik..."):
                try:
                    # Pastikan sesi ada
                    if "chat_session" in st.session_state:
                        response = st.session_state.chat_session.send_message(prompt)
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    else:
                        st.error("Sesi kedaluwarsa. Mohon refresh halaman.")
                        
                except Exception as e:
                    # Penanganan Error KHUSUS jika client terputus
                    error_msg = str(e)
                    if "closed" in error_msg or "client" in error_msg:
                        st.warning("♻️ Koneksi ter-reset. Sedang memuat ulang sesi...")
                        # Hapus sesi yang rusak
                        if "chat_session" in st.session_state:
                            del st.session_state.chat_session
                        # Mulai ulang aplikasi
                        st.rerun()
                    else:
                        st.error(f"Error: {e}")
