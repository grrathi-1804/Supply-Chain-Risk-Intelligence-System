# 📦 Supply Chain Risk Intelligence System

### *Predictive Analytics Engine for Global Logistics*

---

## 🚀 Project Overview
In global trade, late deliveries are a massive financial liability. This project provides a **Production-Ready Machine Learning Solution** to identify high-risk shipments before they leave the warehouse. 

By leveraging the **DataCo Smart Supply Chain Dataset**, this system processes over 144,000 records to deliver real-time risk assessments through a modern web interface.

---

## 🧠 The "Hybrid" Innovation
Unlike standard classification projects, this system utilizes a **Two-Stage Hybrid AI Architecture**:

* **Unsupervised Segmentation (K-Means):** The engine first clusters orders into behavioral "profiles" based on Sales, Quantity, Benefit, and Order Duration. This "pre-digests" complex patterns into distinct segments.
* **Supervised Classification (Logistic Regression):** The segment identity is then fed—alongside temporal and geographical features—into a highly optimized Logistic Regression model.

**The Result:** A highly interpretable model that achieved a **95.5% F1-Score**, outperforming more complex ensemble methods in our benchmarking "Tournament".

---

## 🌟 Key Features
* **Real-time Prediction:** Instant risk assessment using a live Flask backend.
* **Geospatial Awareness:** Analyzes risk based on global markets (Pacific Asia, USCA, etc.) and specific regions.
* **Financial Insight:** Incorporates sales value, discounts, and net totals to determine risk profiles.
* **Glassmorphism UI:** A spectacular, modern dashboard designed for an elite user experience.

---

## 🛠️ Technical Architecture
This project follows a strictly **Modular Production Framework**:

* **Data Ingestion:** Automated pipeline to ingest raw data and perform train-test splits.
* **Data Transformation:** Robust pipeline featuring `ColumnTransformer`, `OneHotEncoding`, and **K-Means Clustering** integration.
* **Model Trainer:** An automated "Tournament" component that benchmarks multiple architectures using `GridSearchCV`.
* **Web Dashboard:** A Flask-based interface utilizing modern design principles.

---

## 📊 Performance Leaderboard
| Model | F1-Score |
| :--- | :--- |
| **Logistic Regression (Winner)** | **0.9558** |
| XGBClassifier | 0.9555 |
| Random Forest | 0.9501 |

---

## 🛤️ Future Roadmap
* **Dockerization:** Containerizing the app for seamless deployment on AWS/Azure.
* **Interactive Analytics:** Adding a Plotly dashboard to show historical risk trends.
* **API Documentation:** Implementing Swagger UI for third-party service integration.

---

## ⚡ Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the Portal**
     ```bash
   python app.py
   ```

## 📁 Project Structure
* **src/components:** Data processing and model training logic.
* **src/pipeline:** Real-time prediction handling.
* **artifacts:** Saved pickle files for the Scaler, K-Means, and Final Model.
* **templates:** High-end Flask UI dashboard.
   
