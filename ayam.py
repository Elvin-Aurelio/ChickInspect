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

if st.checkbox("💬 Konsultasi dengan Dokter AI"):
   st.info("Fitur chatbot akan segera hadir!")