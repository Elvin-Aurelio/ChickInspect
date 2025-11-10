# 🐔 ChikInspect: Poultry Disease Detection using Deep Learning

### *AI-powered image classification for early detection of poultry diseases through fecal analysis.*

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Model Overview & Results](#model-overview--results)
- [Streamlit Application](#streamlit-application)
- [Example Output](#example-output)
- [Technologies Used](#technologies-used)
- [License & Contributors](#license--contributors)

---

## 🧠 Project Overview

**ChikInspect** is a deep learning project designed to assist poultry farmers in detecting common chicken diseases based on fecal images.  
The model utilizes **transfer learning with MobileNetV2**, offering efficient classification performance even on limited computational resources.

### 🎯 Objectives
- Detect and classify poultry fecal images into **Coccidiosis**, **New Castle Disease (ND)**, **Salmonella**, or **Healthy**.
- Provide an accessible **AI-based diagnostic web tool** using **Streamlit**.
- Support smallholder farmers with **actionable recommendations** to minimize loss and improve farm health management.

### 💡 Background
Poultry farming in developing regions often faces challenges such as high feed costs, poor hygiene management, and rapid spread of infectious diseases.  
Diseases like **Newcastle Disease** can cause up to 100% mortality if undetected early.  
By leveraging **AI-driven image classification**, farmers can monitor their flocks proactively without requiring laboratory diagnostics.

---

## 📁 Folder Structure

```

ChikInspect/
│
├── chikinspect_model_A.keras        # Trained MobileNetV2 model
├── ayam.py                          # Streamlit web app for inference
├── notebook.ipynb                   # Colab training & evaluation notebook
├── requirements.txt                 # Python dependencies
├── data.zip                         # (Optional) Original dataset (if stored locally)

```

---

## 📊 Dataset

- **Source:** [Poultry Pathology Visual Dataset (Kaggle)](https://www.kaggle.com/datasets)  
- **Classes Used:**
  - Coccidiosis  
  - Healthy  
  - New Castle Disease  
  - Salmonella
- **Format:** JPEG images (feses ayam)
- **Preprocessing:** Only original images were used.  
  *Augmented images were excluded because many were visually unrealistic and degraded model quality.*

**Dataset Split Summary:**

| Set Type  | Number of Images |
|------------|------------------|
| Training   | 5,592 |
| Validation | 1,399 |
| Testing    | 1,076 |


---

## ⚙️ Installation

### Requirements
- Python 3.9 or higher  
- Recommended: GPU-enabled environment (e.g. Google Colab or local CUDA setup)

### Steps

1. **Clone repository**
   ```bash
   git clone https://github.com/<your-username>/ChikInspect.git
   cd ChikInspect
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Run notebook in Google Colab**
   Make sure to mount Google Drive and place `data.zip` in:

   ```
   /content/drive/MyDrive/data.zip
   ```

---

## 🚀 How to Run

### 🧪 1. Train and Evaluate the Model

Open `notebook.ipynb` and run all cells sequentially.
The notebook includes:

* Data extraction & preprocessing
* Model training (initial + fine-tuning stages)
* Evaluation on test set
* Model saving to `.keras` format

### 💻 2. Run the Streamlit App Locally

After saving your trained model:

```bash
streamlit run ayam.py
```

Then open your browser at `http://localhost:8501`

You can upload a chicken feces image (JPG/PNG), and the app will:

1. Predict disease class
2. Show prediction probabilities
3. Provide actionable disease management recommendations

---

## 📈 Model Overview & Results

**Architecture:**

* Base Model: **MobileNetV2** (pretrained on ImageNet)
* Custom Layers: GlobalAveragePooling → Dropout(0.3) → Dense(4, Softmax)
* Data Augmentation: Random Flip, Rotation, Zoom
* Optimization: Adam (1e-3, fine-tuned 1e-5)
* Class Weights: Applied to handle dataset imbalance

**Classification Report (Test Set, n = 1076)**

| Class              | Precision | Recall | F1-Score | Support |
| ------------------ | --------- | ------ | -------- | ------- |
| Coccidiosis        | 0.99      | 0.93   | 0.96     | 343     |
| Healthy            | 0.87      | 0.93   | 0.90     | 304     |
| New Castle Disease | 0.83      | 0.82   | 0.82     | 77      |
| Salmonella         | 0.93      | 0.93   | 0.93     | 352     |
| **Accuracy**       | **0.92**  |        |          | 1076    |

---

## 🌐 Streamlit Application

The **ChikInspect Web App** provides an interactive tool for AI-based poultry disease diagnosis.

### Features:

* Upload fecal image (auto-resized to 224×224)
* Real-time disease prediction using MobileNetV2
* Confidence probabilities for each of the 4 classes
* **Smart recommendation engine**:

  * 🟥 **High Urgency** → Immediate vet contact & isolation
  * 🟧 **Medium Urgency** → Monitor closely & improve hygiene
  * 🟩 **Low Urgency** → Continue normal observation
* Option to save prediction history (CSV)

### Run command:

```bash
streamlit run ayam.py
```

---

## 🧾 Example Output

```
Prediction: Coccidiosis (95.12% confidence)
Urgency: HIGH

Recommended Actions:
- Isolate suspected birds immediately.
- Replace litter and disinfect the coop.
- Provide supportive care (electrolytes, warmth).
- Consult veterinarian for anticoccidial treatment.
```



---

## 🧰 Technologies Used

| Category               | Tools / Libraries               |
| ---------------------- | ------------------------------- |
| **Deep Learning**      | TensorFlow, Keras               |
| **Data Handling**      | Pandas, NumPy, Scikit-learn     |
| **Visualization**      | Matplotlib, Seaborn             |
| **Deployment**         | Streamlit                       |
| **Hardware**           | Google Colab (GPU T4)           |
| **Model Architecture** | MobileNetV2 (Transfer Learning) |

---

## 📜 License & Contributors

### License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Contributor

**Elvin Aurelio**
*Data Science Student | Machine Learning Enthusiast*

---

> *ChikInspect combines deep learning and social impact — empowering smallholder farmers with affordable, data-driven tools for poultry health and food security.*


