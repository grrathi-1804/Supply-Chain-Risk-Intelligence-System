# 📦 Supply Chain Risk Intelligence System
### *Predictive Analytics Engine for Global Logistics*

---

## 🚀 Project Overview
In global trade, late deliveries aren't just an inconvenience—they are a massive financial liability. This project provides a **Production-Ready Machine Learning Solution** to identify high-risk shipments before they leave the warehouse.

By leveraging the **DataCo Smart Supply Chain Dataset**, this system processes over 144,000 records to deliver real-time risk assessments through a modern web interface.

---

## 🧠 The "Hybrid" Innovation
Unlike standard classification projects, this system utilizes a **Two-Stage Hybrid AI Architecture**:

* **Unsupervised Segmentation (K-Means):** The engine first clusters orders into behavioral "profiles" based on Sales, Quantity, Benefit, and Order Duration. This "pre-digests" complex patterns into distinct segments.
* **Supervised Classification (Logistic Regression):** The segment identity is then fed—alongside temporal and geographical features—into a highly optimized Logistic Regression model.

**The Result:** A highly interpretable model that achieved a **95.5% F1-Score**, outperforming more complex ensemble methods like XGBoost in our benchmarking "Tournament".

---

## 🛠️ Technical Architecture
This project follows a strictly **Modular Production Framework**:

* **Data Ingestion:** Automated pipeline to ingest raw data, perform train-test splits, and handle large-scale CSV processing.
* **Data Transformation:** A robust pipeline featuring `ColumnTransformer`, `OneHotEncoding`, and the integration of our **K-Means Clustering** logic.
* **Model Trainer:** An automated "Tournament" component that benchmarks multiple architectures using `GridSearchCV`.
* **Web Dashboard:** A spectacular Flask-based interface utilizing **Glassmorphism** design principles.

---

## 📊 Performance Leaderboard
| Model | F1-Score |
| :--- | :--- |
| **Logistic Regression (Winner)** | **0.9558** |
| XGBClassifier | 0.9555 |
| Random Forest | 0.9501 |

---

## ⚡ Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
    ```
2.**Launch the portal:**
  ```bash
   python app.py
   ```
