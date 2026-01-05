# ChikInspect – AI-Based Poultry Disease Detection & Consultation System

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://chickinspect-25p28p2ip34udpm8cwffmm.streamlit.app/)


### AI-powered fecal image analysis for early poultry disease detection, enhanced with object detection and AI-assisted veterinary consultation.

---

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Folder Structure](#folder-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Model & Analysis Results](#model--analysis-results)
- [Streamlit Application Usage](#streamlit-application-usage)
- [Example Output](#example-output)
- [Technologies Used](#technologies-used)
- [Limitations & Future Work](#limitations--future-work)
- [License & Contributor](#license--contributor)

---

## Project Overview

**ChikInspect** is an end-to-end AI system designed to assist poultry farmers in **early detection of chicken diseases through fecal image analysis**.  
The project integrates **computer vision, deep learning classification, and a large language model (LLM)** to provide not only predictions, but also **context-aware veterinary consultation**.

### Objectives
- Detect and localize chicken feces in images using object detection.
- Classify detected feces into common poultry disease categories:
  - Coccidiosis
  - Newcastle Disease
  - Salmonella
  - Healthy
- Provide **AI-assisted medical guidance** including treatment suggestions and prevention strategies.
- Deliver the system through an **accessible Streamlit web application**.

### Background
Poultry farming, especially in developing regions, is highly vulnerable to infectious diseases.  
Diseases such as **Newcastle Disease** can cause near-total flock mortality if not detected early.  
Traditional diagnosis requires laboratory testing, which is often inaccessible to smallholder farmers.

ChikInspect addresses this gap by combining **image-based disease detection** with **AI-driven consultation**, enabling proactive and low-cost decision support.

---

## System Architecture

```

Input Image
↓
Roboflow Object Detection (feces localization)
↓
Bounding Box Cropping
↓
CNN Disease Classification (Keras)
↓
Confidence Scoring
↓
Gemini LLM (context-aware consultation)

```

Final disease confidence is computed as:

```

final_score = detection_confidence × classification_confidence

```

---

## Folder Structure

```

ChikInspect/
│
├── ayam.py                               # Streamlit app (detection + chatbot)
├── chikinspect_model_cropped_final.keras # Trained CNN classifier
├── requirements.txt                     # Python dependencies
├── packages.txt                         # Additional environment packages
└── .streamlit/
└── secrets.toml                     # API keys configuration

````

---

## Dataset

- **Source:** Tanzanian Fecal Image Dataset (Zenodo)
- **Data Type:** JPEG images of chicken feces
- **Classes:**
  - Coccidiosis
  - Healthy
  - Newcastle Disease
  - Salmonella

### Dataset Split (Training Phase)

| Set        | Number of Images |
|-----------|------------------|
| Training  | 5,761 |
| Validation| 1,246 |
| Testing   | 1,255 |


---

## Installation

### Requirements
- Python 3.8 or higher
- Pip
- Internet connection (Roboflow & Gemini APIs)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Elvin-Aurelio/ChikInspect.git
   cd ChikInspect
   git checkout yolo
   ```


2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Create a file `.streamlit/secrets.toml`:

```toml
roboflow_api_key = "YOUR_ROBOFLOW_API_KEY"

GEMINI_API_KEY_1 = "YOUR_GEMINI_API_KEY"
# Optional fallback
GEMINI_API_KEY_2 = "YOUR_SECOND_GEMINI_API_KEY"
```

API keys may also be provided via environment variables.

---

## How to Run

Start the Streamlit application:

```bash
streamlit run ayam.py
```

Open your browser at:

```
http://localhost:8501
```

---

## Model & Analysis Results

### Classification Model

* Architecture: CNN (MobileNetV2-based)
* Input size: 224 × 224
* Output: 4-class softmax
* Loss: Categorical Crossentropy
* Optimizer: Adam

### Test Performance (Baseline Model)

| Class             | Precision | Recall | F1-Score |
| ----------------- | --------- | ------ | -------- |
| Coccidiosis       | 0.99      | 0.93   | 0.96     |
| Healthy           | 0.87      | 0.93   | 0.90     |
| Newcastle Disease | 0.83      | 0.82   | 0.82     |
| Salmonella        | 0.93      | 0.93   | 0.93     |
| **Accuracy**      | **0.92**  |        |          |

---

## Streamlit Application Usage

### Workflow

1. Upload an image of chicken feces (JPG/PNG).
2. Click **“Detect Disease”**.
3. The app will:

   * Detect feces using Roboflow.
   * Display bounding boxes.
   * Predict disease and confidence score.
4. Use the **AI consultation chat** to ask about:

   * Treatment options
   * Medication suggestions
   * Prevention strategies

### Features

* Persistent diagnosis display during chat interaction
* Session-based history with CSV export
* Context-aware medical chatbot (Bahasa Indonesia)

---

## Live Demo

The application is publicly available via Streamlit Cloud:

🔗 https://chickinspect-25p28p2ip34udpm8cwffmm.streamlit.app/


## Example Output

```
Primary Diagnosis: Coccidiosis
Confidence Score: 0.87

AI Recommendation:
- Isolate affected chickens.
- Improve coop hygiene.
- Consider anticoccidial treatment such as Amprolium.
- Monitor flock closely for further symptoms.
```

---

## Technologies Used

| Category             | Tools / Libraries              |
| -------------------- | ------------------------------ |
| Deep Learning        | TensorFlow, Keras              |
| Computer Vision      | Roboflow Inference SDK, Pillow |
| Data Processing      | NumPy, Pandas                  |
| AI Assistant         | Google Gemini API              |
| Deployment           | Streamlit                      |
| Training Environment | Google Colab (GPU)             |

---

## Limitations & Future Work

### Current Limitations

* Depends on external APIs (Roboflow, Gemini).
* No probabilistic calibration or uncertainty estimation.
* Session history is not persistent across refresh.

### Future Improvements

* Local object detection model (YOLO inference without API).
* Persistent database for diagnosis history.
* Severity grading and risk stratification.
* Mobile-first deployment.

---

## License & Contributor

### License

This project is licensed under the **MIT License**.

### Contributor

**Elvin Aurelio**
Data Science Student | Applied Machine Learning

---

*ChikInspect aims to bridge AI research and real-world agricultural impact by empowering farmers with accessible, intelligent diagnostic tools.*



---
This project is provided for research and educational purposes only and is not intended as a substitute for professional veterinary diagnosis.
