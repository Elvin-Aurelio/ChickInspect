# ==========================================
# 5. HALAMAN DETEKSI PENYAKIT (DIPERBARUI)
# ==========================================
def render_detection_page():
    st.title("🐔 ChikInspect AI - Deteksi Penyakit")
    st.markdown("Upload foto feses ayam untuk mendeteksi penyakit secara otomatis.")
    
    # --- BAGIAN UPLOAD FILE (SEKARANG DI ATAS) ---
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
                    final_score = pred['confidence'] * disease_conf
                    
                    results.append({
                        "id": i+1,
                        "disease": label,
                        "final_score": final_score,
                        "roboflow_confidence": pred['confidence'],
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
                        st.caption(f"{res['disease']}")
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

    # --- NAVIGATION KE CHATBOT (DIPINDAHKAN KE BAWAH) ---
    st.divider()
    # Opsional: Berikan teks ajakan jika sudah ada hasil diagnosa
    if "prediction_context" in st.session_state:
        st.markdown("##### 💡 Ingin saran pengobatan?")
    
    if st.button("💬 Ke Halaman Chatbot (Konsultasi)", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()
