📦 Supply Chain Risk Intelligence System
Predictive Analytics Engine for Global Logistics
🚀 Project Overview
In global trade, late deliveries aren't just an inconvenience—they are a massive financial liability. This project provides a Production-Ready Machine Learning Solution to identify high-risk shipments before they leave the warehouse.

By leveraging the DataCo Smart Supply Chain Dataset, this system processes over 144,000 records to deliver real-time risk assessments through a modern web interface.

🧠 The "Hybrid" Innovation
Unlike standard classification projects, this system utilizes a Two-Stage Hybrid AI Architecture:

Unsupervised Segmentation (K-Means): The engine first clusters orders into behavioral "profiles" based on Sales, Quantity, Benefit, and Order Duration. This "pre-digests" complex patterns into distinct segments.

Supervised Classification (Logistic Regression): The segment identity is then fed—alongside temporal and geographical features—into a highly optimized Logistic Regression model.

The Result: A highly interpretable model that achieved a 95.5% F1-Score, outperforming more complex ensemble methods like XGBoost in our benchmarking "Tournament".

🛠️ Technical Architecture
This project follows a strictly Modular Production Framework:

Data Ingestion: Automated pipeline to ingest raw data, perform train-test splits, and handle large-scale CSV processing.

Data Transformation: A robust pipeline featuring ColumnTransformer, OneHotEncoding, and the integration of our K-Means Clustering logic.

Model Trainer: An automated "Tournament" component that benchmarks 6 different architectures (XGBoost, Random Forest, etc.) using GridSearchCV for hyperparameter optimization.

Predict Pipeline: A dedicated service layer that bridges the saved .pkl artifacts with real-time user input.

Web Dashboard: A spectacular Flask-based interface utilizing Glassmorphism design principles for an enterprise-grade user experience.

📁 Repository Structure
├── artifacts/           # Trained Models (KMeans, Preprocessor, Logistic Regression)
├── logs/                # Automated system logs for tracking & debugging
├── notebook/            # Exploratory Data Analysis (EDA) & Prototyping
├── src/
│   ├── components/      # Ingestion, Transformation, & Model Training
│   ├── pipeline/        # Prediction Pipeline & Custom Data Mapping
│   ├── exception.py     # Custom Error Handling
│   ├── logger.py        # System Logging
│   └── utils.py         # Common helper functions (Pickling, Evaluation)
├── templates/           # Spectacular Flask HTML Templates
├── app.py               # Flask Web Application
└── requirements.txt     # Reproducible Environment configuration

⚡ Quick Start
1. Clone & Setup
git clone https://github.com/grrathi-1804/Supply-Chain-Risk-Intelligence-System.git
cd Supply-Chain-Risk-Intelligence-System
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows

2. Install Dependencies
pip install -r requirements.txt

3. Launch the Intelligence Portal
python app.py

Open http://127.0.0.1:5000 in your browser to start running risk analyses!

📈 Performance Leaderboard
During the training phase, our "Model Tournament" yielded the following results (F1-Score):
Model,F1-Score
Logistic Regression (Winner),0.9558
XGBClassifier,0.9555
Gradient Boosting,0.9552
Random Forest,0.9501

🤝 Acknowledgments
Special thanks to the modular framework concepts popularized by Krish Naik, which served as the foundation for this production-grade architecture.