import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient
import io
import pandas as pd
from datetime import datetime 

# ==========================================
# 1. KONFIGURASI HALAMAN & CONSTANT
# ==========================================
st.set_page_config(
    page_title="ChikInspect - AI Diagnosis",
    page_icon="🐔",
    layout="wide" # [UBAH] Layout wide agar tabel history muat
)

# Sesuaikan dengan nama file model Anda
MODEL_PATH = 'chikinspect_model_cropped_final.keras' 

# Sesuaikan urutan kelas ini dengan urutan folder saat training (Abjad)
CLASS_NAMES = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# [BARU] Inisialisasi Session State untuk History
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ==========================================
# 2. FUNGSI LOADING MODEL (CACHED)
# ==========================================
@st.cache_resource
def load_classifier_model():
    """Load model Keras sekali saja agar hemat memori."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        return None

model = load_classifier_model()

# ==========================================
# 3. FUNGSI DETEKSI (ROBOFLOW)
# ==========================================
def run_roboflow_detection(image_bytes):
    try:
        api_key = st.secrets["roboflow_api_key"]
    except:
        st.warning("API Key belum disetting di secrets.toml.")
        return None

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    import base64
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        resp = client.run_workflow(
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

# ==========================================
# 4. FUNGSI UTILITY
# ==========================================
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
    img = crop_img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return CLASS_NAMES[class_idx], confidence

# ==========================================
# 5. UI UTAMA (STREAMLIT)
# ==========================================

# --- SIDEBAR HISTORY [BARU] ---
with st.sidebar:
    st.title("📂 Riwayat Diagnosa")
    st.markdown("Daftar hasil pemeriksaan sesi ini:")
    
    if len(st.session_state['history']) > 0:
        # Convert list of dicts to DataFrame
        df_hist = pd.DataFrame(st.session_state['history'])
        
        # Tampilkan tabel ringkas
        st.dataframe(df_hist[['Waktu', 'Diagnosa', 'Skor']], hide_index=True)
        
        # Tombol Download CSV
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Laporan (CSV)",
            data=csv,
            file_name='riwayat_diagnosa_chikinspect.csv',
            mime='text/csv',
        )
        
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state['history'] = []
            st.rerun()
    else:
        st.info("Belum ada data diagnosa.")

# --- MAIN CONTENT ---
st.title("🐔 ChikInspect AI")
st.markdown("Upload foto feses ayam untuk mendeteksi penyakit secara otomatis.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Gambar Asli", use_column_width=True)

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
        
        MIN_DETECTION_CONFIDENCE = 0.5 
        valid_predictions = [p for p in predictions if p['confidence'] >= MIN_DETECTION_CONFIDENCE]

        if not valid_predictions:
            st.warning("⚠️ Objek terdeteksi, namun tingkat keyakinannya terlalu rendah. Mohon foto ulang.")
        else:
            for i, pred in enumerate(valid_predictions):
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                crop_img = original_image.crop((x-w/2, y-h/2, x+w/2, y+h/2))
                
                label, disease_conf = predict_crop(crop_img, model)
                bbox_conf = pred['confidence']
                final_score = bbox_conf * disease_conf
                
                results.append({
                    "id": i+1,
                    "bbox_conf": bbox_conf,
                    "disease": label,
                    "disease_conf": disease_conf,
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
                    st.caption(f"**{res['disease']}** ({res['final_score']:.2f})")

            st.divider()
            
            # --- KESIMPULAN & PENYIMPANAN HISTORY ---
            best_pred = max(results, key=lambda x: x['final_score'])
            
            st.success(f"### ✅ Kesimpulan Diagnosa: {best_pred['disease']}")
            st.markdown(f"Confidence Score: **{best_pred['final_score']:.3f}**")
            
            # [BARU] LOGIKA MENYIMPAN KE HISTORY
            timestamp = datetime.now().strftime("%H:%M:%S")
            new_record = {
                "Waktu": timestamp,
                "Nama File": uploaded_file.name,
                "Objek Terdeteksi": len(valid_predictions),
                "Diagnosa": best_pred['disease'],
                "Skor": round(best_pred['final_score'], 3)
            }
            # Tambahkan ke session state
            st.session_state['history'].append(new_record)
            
            # Tampilkan notifikasi kecil bahwa data tersimpan
            st.toast("✅ Data diagnosa berhasil disimpan ke Riwayat!", icon="💾")

            # Rekomendasi
            with st.expander("ℹ️ Rekomendasi Penanganan Awal", expanded=True):
                if best_pred['disease'] == 'Coccidiosis':
                    st.write("- 🔴 **Urgent:** Pisahkan ayam sakit. Berikan obat anticoccidial (Amprolium).")
                elif best_pred['disease'] == 'New Castle Disease':
                    st.error("💀 **BAHAYA TINGGI!** Isolasi total dan lapor dinas setempat.")
                elif best_pred['disease'] == 'Salmonella':
                    st.write("- Berikan antibiotik sesuai resep. Cek sanitasi air.")
                else:
                    st.write("- ✅ Ayam sehat. Lanjutkan perawatan rutin.")


import streamlit as st
import os
from google import genai
from google.genai.errors import APIError

# --- 0. KONFIGURASI DAN SET UP API KEY ---

# Mengambil API Key dari Environment Variable (GEMINI_API_KEY)
# INI ADALAH CARA AMAN. Kunci API Anda TIDAK tersimpan di file ini.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    st.error("❌ Kesalahan Konfigurasi API: GEMINI_API_KEY belum diset.")
    st.info("⚠️ Silakan set Environment Variable Anda di terminal (misalnya: export GEMINI_API_KEY='...'). Aplikasi dihentikan.")
    st.stop() 

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    st.error("❌ Kesalahan Klien Gemini: Tidak dapat terhubung.")
    st.stop()

# Tentukan peran AI (System Instruction) untuk Dokter AI
SYSTEM_PROMPT = (
    "Anda adalah Dokter AI ahli dalam kesehatan unggas dan diagnosis penyakit ayam. "
    "Fokus utama Anda adalah menganalisis gejala, memberikan saran pencegahan, dan informasi umum. "
    "Ketika diminta menganalisis feses atau gambar, selalu arahkan pengguna untuk menggunakan fitur unggah foto di aplikasi. "
    "Berikan jawaban yang singkat, informatif, dan profesional. Selalu jawab dalam Bahasa Indonesia."
)

# --- 1. SET UP TATA LETAK APLIKASI ---

st.set_page_config(
    page_title="ChikInspect AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar (Riwayat Diagnosa)
st.sidebar.title("📚 Riwayat Diagnosa")
st.sidebar.write("Daftar hasil pemeriksaan sesi ini:")
st.sidebar.info("Belum ada data diagnosa.")

# Judul Utama
st.title("🐔 ChikInspect AI")

# --- 2. FITUR UPLOAD FESES (Diagnosis Otomatis) ---

st.subheader("Upload foto feses ayam untuk mendeteksi penyakit secara otomatis.")

# Area Drop and Drag
uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=False,
    help="Limit 200MB per file. JPG, JPEG, PNG"
)

# Tampilkan gambar dan proses (Saat ini hanya placeholder)
if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    st.image(uploaded_file, caption=uploaded_file.name, width=250)
    st.success(f"File **{uploaded_file.name}** berhasil diunggah.")
    st.info("Proses analisis gambar oleh model ML akan dimulai di sini. Fitur diagnosis sedang dikembangkan.")


st.markdown("---")

# --- 3. FITUR CHATBOT DOKTER AI (Gemini Powered) ---

# Pengganti fungsional dari if st.checkbox("Konsultasi dengan Dokter AI"):
st.header("💬 Konsultasi dengan Dokter AI")
st.caption("Silakan ajukan pertanyaan seputar gejala, pencegahan, atau penyakit ayam...")

# Inisialisasi Riwayat Chat
if "messages" not in st.session_state:
    # Mulai dengan pesan sistem (tidak ditampilkan) dan pesan sambutan
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, 
        {"role": "assistant", "content": "Halo! Saya Dokter AI ChikInspect. Ada yang bisa saya bantu terkait kesehatan ayam Anda?"}
    ]

# Inisialisasi Chat Session Gemini (Untuk mempertahankan konteks percakapan)
if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT
    )

# Menampilkan Riwayat Chat (Lewati pesan pertama yang role: system)
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Pengguna dan Generasi Respon
if prompt := st.chat_input("Tanyakan penyakit, gejala, atau pencegahan..."):
    
    # 1. Tampilkan pesan pengguna
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tambahkan pesan pengguna ke riwayat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Kirim ke Gemini dan dapatkan respons
    with st.chat_message("assistant"):
        with st.spinner("Dokter AI sedang menganalisis..."):
            
            try:
                # Kirim prompt ke chat session untuk mempertahankan konteks
                response = st.session_state.chat_session.send_message(prompt)
                st.markdown(response.text)
                
                # Tambahkan respons asisten ke riwayat
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            
            except APIError:
                st.error("Terjadi kesalahan pada koneksi Gemini API. Pastikan API Key Anda valid dan coba lagi.")
            except Exception:
                st.error("Terjadi kesalahan yang tidak terduga saat memproses permintaan Anda.")

