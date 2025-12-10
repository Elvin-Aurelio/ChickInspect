import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
import io

# ==========================================
# 1. KONFIGURASI HALAMAN & CONSTANT
# ==========================================
st.set_page_config(
    page_title="ChikInspect - AI Diagnosis",
    page_icon="🐔",
    layout="centered"
)

# Sesuaikan dengan nama file model Anda
MODEL_PATH = 'chikinspect_model_cropped_final.keras' 

# Sesuaikan urutan kelas ini dengan urutan folder saat training (Abjad)
CLASS_NAMES = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

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
# 3. FUNGSI DETEKSI (ROBOFLOW) - PERBAIKAN
# ==========================================
def run_roboflow_detection(image_bytes):
    """Mengirim gambar ke Roboflow Workflow API."""
    try:
        api_key = st.secrets["roboflow_api_key"]
    except:
        st.warning("API Key belum disetting di secrets.toml.")
        return []

    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )

    image = Image.open(io.BytesIO(image_bytes))

    try:
        # HAPUS 'use_cache=True' DI SINI
        resp = client.run_workflow(
            workspace_name="elvin-3wtt1",
            workflow_id="find-feses-3",
            images={"image": image} 
        )
        
        if resp and len(resp) > 0:
            # Sesuaikan key output json workflow Anda
            return resp[0].get('predictions', [])
            
    except Exception as e:
        st.error(f"Error Roboflow: {e}")
        return []
    
    return []

# ==========================================
# 4. FUNGSI UTILITY (CROP & DRAW)
# ==========================================
def draw_bounding_boxes(image, predictions):
    """Menggambar kotak di sekitar objek yang terdeteksi."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Coba load font, kalau tidak ada pakai default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for pred in predictions:
        # Konversi Center-XY (Roboflow) ke Corner-XY (PIL)
        # JSON: x, y adalah titik tengah. width, height adalah ukuran.
        x_center, y_center = pred['x'], pred['y']
        w, h = pred['width'], pred['height']
        
        x_min = x_center - (w / 2)
        y_min = y_center - (h / 2)
        x_max = x_center + (w / 2)
        y_max = y_center + (h / 2)

        # Gambar Kotak
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        
        # Tulis Label Confidence Deteksi
        text = f"Feses: {pred['confidence']:.2f}"
        
        # Hitung posisi text background agar rapi
        # bbox return (left, top, right, bottom)
        text_bbox = draw.textbbox((x_min, y_min), text, font=font)
        draw.rectangle((text_bbox[0], text_bbox[1]-5, text_bbox[2], text_bbox[3]+5), fill="red")
        draw.text((x_min, y_min-5), text, fill="white", font=font)

    return img_draw

def predict_crop(crop_img, model):
    """Melakukan prediksi penyakit pada satu potongan gambar."""
    # Resize ke 224x224 sesuai input MobileNetV2
    img = crop_img.resize((224, 224))
    
    # Konversi ke array numpy
    img_array = np.array(img)
    
    # Karena preprocessing (preprocess_input) SUDAH ADA di dalam model (layer),
    # kita tidak perlu normalisasi manual di sini. Cukup expand dims.
    img_array = np.expand_dims(img_array, axis=0) 
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) # Opsional, biasanya output dense sdh softmax
    
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return CLASS_NAMES[class_idx], confidence

# ==========================================
# 5. UI UTAMA (STREAMLIT)
# ==========================================
st.title("🐔 ChikInspect AI")
st.markdown("Upload foto feses ayam untuk mendeteksi penyakit secara otomatis.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Tampilkan Gambar Asli
    image_bytes = uploaded_file.getvalue()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Gambar Asli", use_column_width=True)

    if st.button("🔍 Deteksi Penyakit"):
        with st.spinner('Sedang memindai objek feses (Roboflow)...'):
            # 2. Deteksi Objek
            predictions = run_roboflow_detection(image_bytes)

        if not predictions:
            st.warning("⚠️ Tidak ada objek feses yang terdeteksi. Coba ambil foto lebih dekat.")
        else:
            # Visualisasi Bounding Box
            bbox_image = draw_bounding_boxes(original_image, predictions)
            with col2:
                st.image(bbox_image, caption=f"Terdeteksi {len(predictions)} Objek", use_column_width=True)
            
            # --- PROSES KLASIFIKASI PER OBJEK ---

            st.divider()
            st.subheader("🔬 Hasil Analisis Laboratorium AI")
            
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            
            # Filter: HANYA proses jika Roboflow yakin itu feses minimal 50% (0.5)
            # Ini menyaring batu/jerami yang "agak mirip" feses
            MIN_DETECTION_CONFIDENCE = 0.5 
            
            valid_predictions = [p for p in predictions if p['confidence'] >= MIN_DETECTION_CONFIDENCE]

            if not valid_predictions:
                st.warning("⚠️ Objek terdeteksi, namun tingkat keyakinannya terlalu rendah (kemungkinan bukan feses). Mohon foto ulang.")
            else:
                for i, pred in enumerate(valid_predictions):
                    # Crop logic
                    x_center, y_center = pred['x'], pred['y']
                    w, h = pred['width'], pred['height']
                    
                    left = x_center - (w / 2)
                    top = y_center - (h / 2)
                    right = x_center + (w / 2)
                    bottom = y_center + (h / 2)
                    
                    # Crop gambar
                    crop_img = original_image.crop((left, top, right, bottom))
                    
                    # Prediksi Penyakit
                    label, disease_conf = predict_crop(crop_img, model)
                    bbox_conf = pred['confidence']
                    
                    # --- LOGIKA BARU: COMPOSITE SCORE ---
                    # Kita kalikan keyakinan Roboflow x Keyakinan MobileNet
                    final_score = bbox_conf * disease_conf
                    
                    results.append({
                        "id": i+1,
                        "bbox_conf": bbox_conf,       # Yakin ini feses?
                        "disease": label,
                        "disease_conf": disease_conf, # Yakin ini penyakit X?
                        "final_score": final_score,   # Skor Akhir (Perkalian)
                        "img": crop_img
                    })
                    progress_bar.progress((i + 1) / len(valid_predictions))
                
                progress_bar.empty()

                # --- TAMPILKAN HASIL INDIVIDUAL ---
                cols = st.columns(min(len(results), 3))
                for idx, res in enumerate(results):
                    with cols[idx % 3]:
                        st.image(res['img'], caption=f"Objek #{res['id']}")
                        # Tampilkan detail skor agar user paham
                        st.markdown(f"**{res['disease']}**")
                        st.caption(f"YOLO: {res['bbox_conf']:.2f} | Model: {res['disease_conf']:.2f}")
                        st.caption(f"**Score: {res['final_score']:.3f}**")

                # --- KESIMPULAN BERDASARKAN FINAL SCORE TERTINGGI ---
                st.divider()
                
                # Cari prediksi dengan FINAL SCORE tertinggi (bukan cuma disease conf)
                best_pred = max(results, key=lambda x: x['final_score'])
                
                st.success(f"### ✅ Kesimpulan Diagnosa: {best_pred['disease']}")
                st.markdown(f"""
                Sistem memilih hasil ini karena memiliki kombinasi keyakinan tertinggi:
                - Tingkat keyakinan objek feses: **{best_pred['bbox_conf']*100:.1f}%**
                - Tingkat kecocokan gejala penyakit: **{best_pred['disease_conf']*100:.1f}%**
                """)
                
                # Rekomendasi Penanganan
                with st.expander("ℹ️ Rekomendasi Penanganan Awal"):
                    if best_pred['disease'] == 'Coccidiosis':
                        st.write("- 🔴 **Urgent:** Pisahkan ayam yang sakit segera.")
                        st.write("- Berikan obat anticoccidial (seperti Amprolium).")
                        st.write("- Jaga kekeringan kandang, ganti sekam yang basah.")
                    elif best_pred['disease'] == 'New Castle Disease':
                        st.error("💀 **BAHAYA TINGGI!** Segera lapor dokter hewan/dinas setempat.")
                        st.write("- Isolasi total area kandang.")
                        st.write("- Vaksinasi darurat untuk ayam yang masih sehat.")
                    elif best_pred['disease'] == 'Salmonella':
                        st.write("- Berikan antibiotik spektrum luas sesuai resep vet.")
                        st.write("- Cek kualitas air minum & sanitasi tempat pakan.")
                    else: # Healthy
                        st.write("- ✅ Kondisi pencernaan ayam tampak normal.")
                        st.write("- Lanjutkan program pakan dan vitamin rutin.")

# Placeholder untuk Chatbot (Next Step)
if st.checkbox("💬 Konsultasi dengan Dokter AI"):
   st.info("Fitur chatbot akan segera hadir!")