# Network Threat Analyzer

A Streamlit-based application for advanced network traffic analysis, attack classification, and anomaly detection using state-of-the-art machine learning and deep learning models.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture & Workflow](#architecture--workflow)
- [Model Details](#model-details)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Sample Results](#sample-results)
- [Credits](#credits)

---

## Overview

**Network Threat Analyzer** is a modular, user-friendly tool for:
- **Classifying network traffic** (e.g., VPN types, attack categories)
- **Detecting anomalies** in network traffic

It leverages classical ML (Random Forest, XGBoost) and deep learning (CNN, NIN) models, trained on real-world datasets, to provide actionable insights for cybersecurity analysis.

---

## Features
- **Streamlit Web UI**: Intuitive, interactive interface for uploading data and viewing results
- **Attack Classification**: Predicts traffic category using XGBoost and deep learning models
- **Anomaly Detection**: Detects network anomalies using a Random Forest model
- **Downloadable Results**: Export predictions as CSV
- **Supports Large Datasets**: Efficient preprocessing and model inference

---

## Project Structure

<details>
<summary><strong>Click to expand project tree</strong></summary>

```plaintext
📦 intel-unnati-models/
 ├── 📄 app.py                  # Main Streamlit app (UI, workflow, model inference)
 ├── 📄 README.md               # Project documentation
 ├── 📁 network-anomaly/        # Anomaly detection resources
 │    ├── 📓 anomaly.ipynb         # Jupyter notebook for anomaly detection model training
 │    ├── 🏷️ model_plain_rf.pkl     # Trained Random Forest model for anomaly detection
 │    └── 🏷️ scaler_plain.pkl        # Scaler for anomaly detection features
 └── 📁 traffic-classification/ # Traffic classification resources
      ├── 📓 classification-model.ipynb # Jupyter notebook for classification model training
      ├── 🏷️ cnn_model.keras           # Trained 1D-CNN model for classification
      ├── 🏷️ nin_model.keras           # Trained NIN-style CNN model for classification
      ├── 🏷️ xgb_model.json            # Trained XGBoost model for classification
      ├── 🏷️ rf_model.joblib           # Trained Random Forest model for classification (not used in app)
      └── 📄 scenario_a_combined.csv    # Combined dataset for VPN/non-VPN classification
```
</details>

### Directory & File Descriptions
- <code>app.py</code>: The main entry point. Implements the Streamlit UI, handles file uploads, loads models/scalers, and runs inference for both classification and anomaly detection.
- <code>network-anomaly/</code>: Contains all resources for anomaly detection:
  - <code>anomaly.ipynb</code>: Notebook for data prep, training, and evaluation of the anomaly detection model.
  - <code>model_plain_rf.pkl</code>: Serialized Random Forest model for anomaly detection.
  - <code>scaler_plain.pkl</code>: Scaler used to normalize features for anomaly detection.
- <code>traffic-classification/</code>: Contains all resources for traffic classification:
  - <code>classification-model.ipynb</code>: Notebook for data prep, training, and evaluation of classification models (Random Forest, XGBoost, CNN, NIN).
  - <code>cnn_model.keras</code>, <code>nin_model.keras</code>: Deep learning models for classification.
  - <code>xgb_model.json</code>: XGBoost model for classification (used in app).
  - <code>rf_model.joblib</code>: Random Forest model for classification (for comparison, not used in app).
  - <code>scenario_a_combined.csv</code>: Main dataset for VPN/non-VPN and traffic type classification.

---

## Architecture & Workflow

![Architecture Diagram](image.png)

### High-Level Flow
1. **User Interface**: The user interacts with the Streamlit web app (`app.py`).
2. **Task Selection**: User selects either Attack Classification or Anomaly Detection from the sidebar.
3. **File Upload**: User uploads a CSV file (format depends on the selected task).
4. **Preprocessing**:
   - For classification: Numeric feature selection, scaling, label encoding (if needed).
   - For anomaly detection: Numeric feature selection, scaling, dropping label columns.
5. **Model Inference**:
   - For classification: XGBoost model (and label encoder) is loaded and used for prediction.
   - For anomaly detection: Random Forest model and scaler are loaded and used for prediction.
6. **Result Mapping**: Model outputs are mapped to human-readable class or attack labels.
7. **Output**: Results are displayed in the UI and can be downloaded as a CSV.

### Data & Model Flow (Step-by-Step)
- **Attack Classification**:
  1. User uploads a CSV with features matching `scenario_a_combined.csv`.
  2. Data is preprocessed (numeric columns, scaling).
  3. XGBoost model (`xgb_model.json`) and label encoder (fit on `scenario_a_combined.csv`) are loaded.
  4. Model predicts class indices, which are mapped to class names (e.g., 'Anonymizing VPN', 'Non-VPN', etc.).
  5. Results are shown and can be downloaded.
- **Anomaly Detection**:
  1. User uploads a CSV (no headers) matching the UNSW-NB15 format.
  2. Data is preprocessed (drops last two columns, numeric selection, scaling).
  3. Random Forest model (`model_plain_rf.pkl`) and scaler (`scaler_plain.pkl`) are loaded.
  4. Model predicts attack categories (e.g., 'DoS', 'Backdoor', 'Normal', etc.).
  5. Results are shown and can be downloaded.

### Model Training (Offline, via Notebooks)
- **traffic-classification/classification-model.ipynb**: Trains and evaluates Random Forest, XGBoost, CNN, and NIN models on VPN/non-VPN and traffic type data. Saves models for use in the app.
- **network-anomaly/anomaly.ipynb**: Trains and evaluates a Random Forest model for anomaly detection on the UNSW-NB15 dataset. Saves model and scaler for use in the app.

---

## Model Details

### **Classification Models**
- **Random Forest**: Baseline classical ML model
- **XGBoost**: High-performance gradient boosting, trained on GPU
- **1D-CNN**: Deep learning model for sequential data
- **NIN-like CNN**: Network-in-Network style deep model

### **Anomaly Detection Model**
- **Random Forest**: Trained on UNSW-NB15 dataset, detects various attack types

### **Training Notebooks**
- `traffic-classification/classification-model.ipynb`: Details data prep, training, evaluation for classification
- `network-anomaly/anomaly.ipynb`: Details data prep, training, evaluation for anomaly detection

---

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd intel-unnati-models
   ```
2. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn xgboost tensorflow joblib
   ```
3. **(Optional) GPU Support**
   - For XGBoost and TensorFlow, install GPU versions if available.

4. **Ensure model files are present** in the correct directories (see Project Structure).

---

## Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
2. **Select a page** in the sidebar:
   - **Attack Classification**: Upload a CSV with features matching the training data (see `scenario_a_combined.csv`)
   - **Anomaly Detection**: Upload a CSV (no headers) matching the UNSW-NB15 format
3. **View results** in the browser and download predictions as CSV.

---

## Sample Results

- **Attack Classification**: Predicts categories like 'Anonymizing VPN', 'Commercial VPN', 'Non-VPN', etc.
- **Anomaly Detection**: Predicts categories like 'DoS', 'Backdoor', 'Normal', etc.

---

## Credits

- Developed as part of the Intel Unnati program
- Uses open-source libraries: Streamlit, scikit-learn, XGBoost, TensorFlow, Pandas, NumPy
- Datasets: VPN/Non-VPN traffic, UNSW-NB15

---

## License

This project is for educational and research purposes. Please cite appropriately if used in academic work.
