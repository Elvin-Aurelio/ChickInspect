import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# ======== CONFIGURATIONS ========
st.set_page_config(page_title="ChikInspect", layout="centered")

st.title("🐔 ChikInspect - Poultry Health Detection")
st.caption("AI-powered fecal image analysis for early detection of poultry diseases")

# ======== LOAD MODEL ========
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "chikinspect_model_A.keras")
    st.write(f"📂 Loading model from: `{model_path}`")  # opsional (boleh dihapus setelah testing)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()
CLASS_NAMES = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
IMG_SIZE = 224

# ======== FILE UPLOAD ========
uploaded_file = st.file_uploader("Upload fecal image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ======== PREPROCESS (NO preprocess_input, NO division) ========
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype('float32')  # raw pixel scale (0–255)
    img_batch = np.expand_dims(img_array, axis=0)

    # ======== PREDICT ========
    with st.spinner("Analyzing image..."):
        predictions = model.predict(img_batch)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = predictions[predicted_index] * 100

    # ======== OUTPUT ========
    st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")

    st.markdown("### Probability per class:")
    for name, prob in zip(CLASS_NAMES, predictions):
        st.write(f"- **{name}**: {prob*100:.2f}%")

    # ======== BAR CHART ========
    chart_data = pd.DataFrame({
        "Class": [name for name in CLASS_NAMES],
        "Probability (%)": [p * 100 for p in predictions]
    })
    st.bar_chart(chart_data, x="Class", y="Probability (%)")


    # ======== RECOMMENDATION LOGIC ========
def get_urgency(prob):
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"

def get_recommendation(pred_class, prob):
    urgency = get_urgency(prob)
    rec = {"title": pred_class, "urgency": urgency, "actions": [], "notes": []}

    if pred_class.lower().startswith("coccid"):
        rec["actions"] = [
            "Isolate suspected birds immediately.",
            "Remove and replace litter; disinfect the coop.",
            "Provide supportive care (electrolytes, warmth).",
            "Consult veterinarian for anticoccidial treatment or prescription."
        ]
        rec["notes"] = [
            "Coccidiosis often related to poor hygiene and contaminated feed/water.",
            "Consider reviewing litter management and vaccination program."
        ]

    elif pred_class.lower().startswith("healthy"):
        rec["actions"] = [
            "Continue routine monitoring (check feces, feed intake, behaviour).",
            "Record this result in farm log."
        ]
        rec["notes"] = [
            "No immediate action required, but maintain good biosecurity and nutrition."
        ]

    elif "new" in pred_class.lower() or "nd" in pred_class.lower():
        rec["actions"] = [
            "Strictly isolate the affected flock/house.",
            "Stop movement of birds, products, and equipment.",
            "Contact local veterinarian and report to livestock authority if required.",
            "Implement urgent biosecurity: disinfect boots, equipment, restrict access."
        ]
        rec["notes"] = [
            "Newcastle Disease can be highly contagious and cause high mortality.",
            "Follow vet guidance for culling or targeted treatment if recommended."
        ]

    elif "salmonella" in pred_class.lower():
        rec["actions"] = [
            "Isolate suspected birds and practice strict hygiene.",
            "Avoid handling eggs/meat without protection—Salmonella is zoonotic.",
            "Consult veterinarian for testing and possible antibiotic or management plan."
        ]
        rec["notes"] = [
            "Investigate feed and water sources, and sanitize feeding equipment.",
            "Cook or handle animal products safely to prevent spread to humans."
        ]

    # Tailor urgency-specific prompt
    if rec["urgency"] == "HIGH":
        rec["priority_note"] = "URGENT: Contact a veterinarian immediately and restrict movements."
    elif rec["urgency"] == "MEDIUM":
        rec["priority_note"] = "Monitor closely and prepare to escalate (collect samples, contact vet)."
    else:
        rec["priority_note"] = "Low urgency: continue routine monitoring."

    return rec

# usage (after predictions computed)
top_prob = float(predictions[predicted_index])
recommendation = get_recommendation(predicted_class, top_prob)

st.markdown("## Recommendation")
st.write(f"**Detected:** {recommendation['title']}")
st.write(f"**Urgency level:** {recommendation['urgency']}")
st.info(recommendation['priority_note'])

st.markdown("### Recommended Actions")
for a in recommendation['actions']:
    st.write(f"- {a}")

if recommendation.get("notes"):
    st.markdown("### Notes / Context")
    for n in recommendation['notes']:
        st.write(f"- {n}")
st.markdown("---")
st.caption("⚠️ Disclaimer: This tool provides AI-based analysis and recommendations. Always consult a qualified veterinarian for definitive diagnosis and treatment.")
# Optional: Save recommendation + prediction to local CSV
if st.button("Save result & recommendation"):
    import json, datetime, csv
    fname = "history_results.csv"
    row = {
        "time": datetime.datetime.utcnow().isoformat(),
        "filename": uploaded_file.name if uploaded_file is not None else "",
        "predicted_class": recommendation['title'],
        "probability": top_prob,
        "urgency": recommendation['urgency'],
        "actions": "; ".join(recommendation['actions'])
    }
    # write header if new
    write_header = not os.path.exists(fname)
    with open(fname, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    st.success(f"Saved to {fname}")

else:
    st.info("Please upload a chicken feces image to start the analysis.")
