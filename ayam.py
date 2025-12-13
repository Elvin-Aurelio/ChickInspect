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

import os

# ==========================================
# 1. KONFIGURASI HALAMAN & API KEY
# ==========================================
st.set_page_config(
    page_title="ChikInspect - AI Diagnosis",
    page_icon="🐔",
    layout="wide"
)

# --- SETUP GEMINI (CHATBOT) ---
# Baca API Key pertama (utama)
GEMINI_API_KEY_1 = os.environ.get('GEMINI_API_KEY_1')
if not GEMINI_API_KEY_1 and "GEMINI_API_KEY_1" in st.secrets:
    GEMINI_API_KEY_1 = st.secrets["GEMINI_API_KEY_1"]

# Baca API Key kedua (cadangan)
GEMINI_API_KEY_2 = os.environ.get('GEMINI_API_KEY_2')
if not GEMINI_API_KEY_2 and "GEMINI_API_KEY_2" in st.secrets:
    GEMINI_API_KEY_2 = st.secrets["GEMINI_API_KEY_2"]

# Untuk backward compatibility, jika masih pakai GEMINI_API_KEY lama
if not GEMINI_API_KEY_1:
    GEMINI_API_KEY_1 = os.environ.get('GEMINI_API_KEY')
    if not GEMINI_API_KEY_1 and "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY_1 = st.secrets["GEMINI_API_KEY"]

# Instruksi Dokter AI
SYSTEM_PROMPT = (
    "Anda adalah Dokter AI ahli kesehatan unggas. "
    "Analisis gejala, berikan saran obat (misal: Amprolium untuk Koksidiosis), dan pencegahan. "
    "Jawab singkat, padat, dan profesional dalam Bahasa Indonesia."
)

# --- SETUP MODEL KLASIFIKASI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chikinspect_model_cropped_final.keras")
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
# 3. FUNGSI-FUNGSI CHATBOT
# ==========================================
def build_system_prompt():
    base = SYSTEM_PROMPT
    if "prediction_context" in st.session_state:
        base += "\n\n" + st.session_state.prediction_context
    return base

def ask_gemini(messages):
    # Build contents - hanya user dan model/assistant, tanpa system
    # Format: Content object dengan role dan parts (list of Part objects)
    contents = []
    for m in messages:
        if m["role"] != "system" and m["content"] != "mengetik...":
            # Convert "assistant" role to "model" for Gemini API
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": m["content"]}]
            })

    # System instruction dikirim via config, bukan contents
    system_prompt = build_system_prompt()
    config = {
        "system_instruction": {"parts": [{"text": system_prompt}]}
    }

    # Coba API Key pertama (utama)
    api_keys_to_try = []
    if GEMINI_API_KEY_1:
        api_keys_to_try.append(("GEMINI_API_KEY_1", GEMINI_API_KEY_1))
    if GEMINI_API_KEY_2:
        api_keys_to_try.append(("GEMINI_API_KEY_2", GEMINI_API_KEY_2))

    if not api_keys_to_try:
        st.error("⚠️ Tidak ada API Key Gemini yang dikonfigurasi. Harap set GEMINI_API_KEY_1 atau GEMINI_API_KEY_2.")
        return "Maaf, API Key tidak tersedia. Silakan hubungi administrator."

    last_error = None
    for api_key_name, api_key in api_keys_to_try:
        try:
            client = genai.Client(api_key=api_key)
            
            # Gunakan gemini-2.5-flash
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=config
            )

            # Jika berhasil, simpan info API key yang digunakan
            if api_key_name != api_keys_to_try[0][0]:
                st.success(f"✅ Menggunakan {api_key_name} (API utama sudah habis)")

            return response.text

        except APIError as e:
            error_msg = str(e).lower()
            last_error = e
            
            # Cek apakah error terkait quota/limit
            if "quota" in error_msg or "limit" in error_msg or "rate limit" in error_msg or "429" in str(e):
                # Jika masih ada API key lain, coba yang berikutnya
                if api_key != api_keys_to_try[-1][1]:
                    st.warning(f"⚠️ {api_key_name} quota habis, mencoba API key cadangan...")
                    continue
                else:
                    # Semua API key sudah habis
                    error_message = (
                        "⚠️ **Semua API Key Quota Terlampaui**\n\n"
                        "Semua quota API Gemini Anda telah habis. "
                        "Silakan:\n"
                        "1. Cek quota di Google AI Studio (https://aistudio.google.com)\n"
                        "2. Tunggu hingga quota direset (biasanya per hari/per bulan)\n"
                        "3. Atau upgrade ke paket berbayar jika perlu\n\n"
                        f"Detail error: {str(e)}"
                    )
                    st.error(error_message)
                    return "Maaf, semua quota API telah habis. Silakan coba lagi nanti atau hubungi administrator."
            else:
                # Error selain quota, langsung return
                st.error(f"Error API Gemini ({api_key_name}): {str(e)}")
                return f"Maaf, terjadi kesalahan: {str(e)}"

        except Exception as e:
            error_msg = str(e).lower()
            last_error = e
            
            # Cek apakah error terkait quota/limit
            if "quota" in error_msg or "limit" in error_msg or "429" in error_msg:
                # Jika masih ada API key lain, coba yang berikutnya
                if api_key != api_keys_to_try[-1][1]:
                    st.warning(f"⚠️ {api_key_name} quota habis, mencoba API key cadangan...")
                    continue
                else:
                    error_message = (
                        "⚠️ **Semua API Key Quota Terlampaui**\n\n"
                        "Semua quota API Gemini Anda telah habis. "
                        "Silakan cek quota di Google AI Studio atau tunggu hingga direset.\n\n"
                        f"Detail: {str(e)}"
                    )
                    st.error(error_message)
                    return "Maaf, semua quota API telah habis. Silakan coba lagi nanti."
            else:
                # Error lain, langsung return
                st.error(f"Error saat memanggil Gemini API ({api_key_name}): {str(e)}")
                return f"Maaf, terjadi kesalahan: {str(e)}"

    # Jika semua API key gagal (tidak masuk kategori di atas)
    if last_error:
        st.error(f"Error saat memanggil Gemini API: {str(last_error)}")
        return f"Maaf, terjadi kesalahan: {str(last_error)}"
    else:
        return "Maaf, terjadi kesalahan yang tidak diketahui."

# ==========================================
# 4. SIDEBAR (SHARED)
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.title("📂 Riwayat Diagnosa")
        st.info("⚠️ Data disimpan sementara di browser Anda. Data akan hilang jika halaman di-refresh.")
        st.markdown("Daftar hasil pemeriksaan sesi ini:")
        
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

# ==========================================
# 5. HALAMAN DETEKSI PENYAKIT
# ==========================================
def render_detection_page():
    st.title("🐔 ChikInspect AI - Deteksi Penyakit")
    st.markdown("Upload foto feses ayam untuk mendeteksi penyakit secara otomatis.")
    
    # Tombol navigasi ke halaman chat
    if st.button("💬 Ke Halaman Chatbot", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()
    
    st.divider()

    # UPLOAD FILE
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    # Reset chat dan context ketika file baru di-upload
    if uploaded_file is not None:
        # Cek apakah ini file baru (berbeda dari sebelumnya)
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            # Reset chat messages dan prediction context
            if "messages" in st.session_state:
                st.session_state.messages = []
            if "prediction_context" in st.session_state:
                del st.session_state.prediction_context
            st.session_state.last_uploaded_file = uploaded_file.name

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
                # Hapus prediction_context jika tidak ada hasil
                if "prediction_context" in st.session_state:
                    del st.session_state.prediction_context
                # Reset chat messages jika tidak ada hasil
                if "messages" in st.session_state:
                    st.session_state.messages = []
            else:
                for i, pred in enumerate(valid_predictions):
                    x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                    crop_img = original_image.crop((x-w/2, y-h/2, x+w/2, y+h/2))
                    
                    label, disease_conf = predict_crop(crop_img, model)
                    roboflow_conf = pred['confidence']
                    final_score = roboflow_conf * disease_conf
                    
                    results.append({
                        "id": i+1,
                        "disease": label,
                        "final_score": final_score,
                        "roboflow_confidence": roboflow_conf,
                        "disease_confidence": disease_conf,
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
                        st.caption(f"Final Score: {res['final_score']:.3f}")
                        st.caption(f"Roboflow: {res['roboflow_confidence']:.3f} | Penyakit: {res['disease_confidence']:.3f}")

                # Kesimpulan & Simpan
                if results:
                    best_pred = max(results, key=lambda x: x['final_score'])

                    st.session_state.prediction_context = f"""
                    Hasil analisis AI terbaru:
                    - Penyakit terdeteksi: {best_pred['disease']}
                    - Final Score: {best_pred['final_score']:.3f}
                    - Roboflow Detection: {best_pred['roboflow_confidence']:.3f}
                    - Disease Classification: {best_pred['disease_confidence']:.3f}
                    - Waktu pemeriksaan: {datetime.now().strftime('%H:%M:%S')}
                    """
                    st.success(f"### ✅ Diagnosa Utama: {best_pred['disease']}")
                    st.info(f"**Final Score:** {best_pred['final_score']:.3f} ({best_pred['final_score']*100:.1f}%)")
                    st.info(f"**Detail Skor:** Roboflow Detection = {best_pred['roboflow_confidence']:.3f} ({best_pred['roboflow_confidence']*100:.1f}%) × Disease Classification = {best_pred['disease_confidence']:.3f} ({best_pred['disease_confidence']*100:.1f}%)")
                    
                    # Simpan ke History
                    st.session_state['history'].append({
                        "Waktu": datetime.now().strftime("%H:%M:%S"),
                        "Nama File": uploaded_file.name,
                        "Diagnosa": best_pred['disease'],
                        "Skor": round(best_pred['final_score'], 3)
                    })
                    st.toast("Data tersimpan!", icon="💾")

# ==========================================
# 6. HALAMAN CHATBOT
# ==========================================
def render_chat_page():
    st.title("💬 Konsultasi dengan Dokter AI")
    st.caption("Diskusikan hasil diagnosa atau tanya tips perawatan ayam...")
    
    # Tombol navigasi ke halaman deteksi
    if st.button("🔍 Ke Halaman Deteksi", use_container_width=True):
        st.session_state.page = "deteksi"
        st.rerun()
    
    st.divider()

    if not GEMINI_API_KEY_1 and not GEMINI_API_KEY_2:
        st.warning("⚠️ Fitur Chatbot belum aktif. Masukkan GEMINI_API_KEY_1 atau GEMINI_API_KEY_2 di Secrets/Environment.")
    else:
        # Inisialisasi messages
        if "messages" not in st.session_state or len(st.session_state.messages) == 0:
            # Cek apakah ada hasil analisis yang sudah dilakukan
            if "prediction_context" in st.session_state and st.session_state.prediction_context:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Halo! Saya sudah melihat hasil analisis ayam Anda. Ada yang ingin ditanyakan?"}
                ]
            else:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Halo! Saya adalah Dokter AI untuk kesehatan unggas. Silakan upload dan deteksi gambar feses ayam terlebih dahulu, atau langsung tanyakan apa yang ingin Anda ketahui tentang kesehatan ayam."}
                ]

        # Cek apakah ada placeholder "mengetik..."
        typing_exists = any(msg["content"] == "mengetik..." for msg in st.session_state.messages)
        
        # Proses jika ada placeholder "mengetik..." (langsung proses, jangan tunggu input)
        if typing_exists:
            # Ambil messages tanpa placeholder untuk API
            messages_for_api = [msg for msg in st.session_state.messages if msg["content"] != "mengetik..."]
            
            # Proses dengan Gemini
            answer = ask_gemini(messages_for_api)
            
            # Hapus placeholder dan tambahkan jawaban sebenarnya
            st.session_state.messages = [msg for msg in st.session_state.messages if msg["content"] != "mengetik..."]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

        # Tampilkan semua messages (termasuk yang akan diproses)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["content"] == "mengetik...":
                    st.markdown("mengetik...")
                else:
                    st.markdown(msg["content"])

        # Input dari user
        if prompt := st.chat_input("Tulis pertanyaan Anda..."):
            # Tambahkan pesan user ke messages
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Tambahkan placeholder "mengetik..." untuk assistant
            st.session_state.messages.append({"role": "assistant", "content": "mengetik..."})
            st.rerun()

# ==========================================
# 7. MAIN APP ROUTING
# ==========================================
# Inisialisasi page
if "page" not in st.session_state:
    st.session_state.page = "deteksi"

# Render sidebar (shared)
render_sidebar()

# Render halaman sesuai state
if st.session_state.page == "deteksi":
    render_detection_page()
elif st.session_state.page == "chat":
    render_chat_page()
