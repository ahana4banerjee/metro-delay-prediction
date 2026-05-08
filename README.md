# 🚇 Smart Delay Prediction in Hyderabad Metro Systems

### Using Hybrid Machine Learning and Operational Insights

> A machine learning framework to predict Hyderabad Metro train delays using GTFS schedule data, realistic delay simulation, operational insights, and cross-validation on Indian Railways data.

-----

**Live Demo:** [https://metro-delay-prediction.streamlit.app](https://metro-delay-prediction.streamlit.app/)

## 📌 Overview

Metro systems depend heavily on punctuality. Even small delays can propagate across downstream stations, impacting schedules, passenger trust, and operational efficiency.

This project builds a **delay prediction system** for Hyderabad Metro using:

  * **GTFS schedule data** as the foundation
  * **Delay simulation** (modeling realistic scenarios where public data is unavailable)
  * **Machine Learning models** (Regression and Time-Series)
  * **Hybrid operational logic** for propagation tracking
  * **Interactive Streamlit dashboard** for real-time visualization
  * **External validation** using Indian Railways dataset

The project demonstrates how data-driven methods can support smarter urban transit systems.

-----

## 🎯 Objectives

  * Predict train delays at station/time level.
  * Identify and understand key factors causing delays (Peak hours, Congestion).
  * Compare performance across multiple ML models (XGBoost, Random Forest, LSTM).
  * Validate model robustness on real-world Indian Railways data.
  * Provide a user-friendly dashboard for operational decision support.

-----

## 🗂 Datasets

### 1\. Hyderabad Metro GTFS Dataset

Structured transit schedule data containing:

  * `agency.txt`, `routes.txt`, `trips.txt`
  * `stop_times.txt`, `stops.txt`, `calendar.txt`
  * `shapes.txt`, `fare_rules.txt`, `fare_attributes.txt`, `feed_info.txt`

### 2\. Indian Railways Delay Dataset

Used for external validation to test the generalization of models on real-world delay behavior.

-----

## ⚙️ Hybrid Prediction Framework

The proposed framework combines:

- Statistical delay simulation
- Ensemble machine learning
- Sequential operational logic
- Explainable AI analysis
- Real-world validation

The system is designed as a hybrid operational intelligence pipeline rather than a standalone prediction model.

### Framework Components

1. GTFS-based metro schedule processing
2. Realistic delay simulation engine
3. Advanced feature engineering
4. Hybrid ensemble prediction model
5. SHAP-based explainability analysis
6. Interactive operational dashboard
7. External validation using Indian Railways data

-----

### 🤖 Hybrid Ensemble Model

Instead of relying on a single prediction algorithm, the project uses a hybrid ensemble strategy combining:

- XGBoost
- Random Forest
- Sequential operational features
- Propagation-aware logic

The hybrid architecture combines machine learning outputs with operational intelligence factors such as:

- Peak-hour congestion
- Station crowding
- Delay propagation
- Sequential station dependency

This improves both prediction robustness and interpretability.

### Why Hybrid Modeling?

Traditional ML models focus only on statistical relationships.  
This framework additionally incorporates transportation-domain operational behavior, making predictions more realistic and practically usable for metro systems.

---

## 🤖 Model Performance

### 📊 Metro Dataset Results

| Model | RMSE | MAE | $R^2$ |
| :--- | :--- | :--- | :--- |
| **XGBoost** | **1.144** | **0.829** | **0.940** |
| **Hybrid (XGB+OAS)** | **0.945** | **0.703** | **0.964** |
| Random Forest | 1.159 | 0.841 | 0.938 |
| Linear Regression | 1.325 | 0.976 | 0.913 |
| LSTM | 1.352 | 1.025 | 0.920 |


### 🚆 Indian Railways Validation

| Model | RMSE | MAE | $R^2$ |
| :--- | :--- | :--- | :--- |
| Random Forest | 7.16 | 3.21 | 0.97 |
| **XGBoost** | **8.13** | **2.97** | **0.95** |
| **Hybrid (XGB+OAS)** | **8.14** | **4.68** | **0.96** |
| LSTM | 13.86 | 6.86 | 0.88 |
| Linear Regression | 15.66 | 7.81 | 0.85 |

-----

## 🧠 Explainable AI using SHAP

To improve transparency and interpretability, SHAP (SHapley Additive exPlanations) analysis was used to understand how features influence delay predictions.

### SHAP Insights

The explainability analysis revealed:

- `Previous Delay` is the dominant contributor to downstream delays
- `Peak Hours` strongly increase delay probability
- `Congestion Level` significantly affects station dwell time
- `Stop Sequence` captures propagation behavior across routes

### Why SHAP Matters

Instead of treating the model as a black box, SHAP provides operational interpretability by showing:

- Why a delay was predicted
- Which features contributed most
- How operational conditions influence predictions

This makes the framework more suitable for real-world transportation decision support systems.

---
## 🔍 Key Research Findings

### Operational Findings

- Delay propagation is the strongest behavioral characteristic in metro systems
- Interchange stations show consistently higher delay accumulation
- Peak-hour congestion significantly amplifies downstream delays
- Delay behaves as a sequential path-dependent process

### Machine Learning Findings

- Ensemble models outperform standalone regression approaches
- XGBoost and Random Forest achieved strongest generalization capability
- Hybrid operational modeling improves realism compared to pure statistical prediction
- Explainable AI methods improve interpretability of transportation ML systems

### Validation Findings

- Models trained on simulated metro logic successfully generalized to real Indian Railways operational data
- Tree-based ensemble models remained robust across structurally different datasets

-----

## 💻 Dashboard Features

### 🌐 Try the Live App
https://metro-delay-prediction.streamlit.app/

Built using **Streamlit**, the dashboard provides:

  * **User Inputs:** Source station, Destination, Day, and Time.
  * **Real-time Predictions:** Predicted delay and estimated arrival time.
  * **Operational Alerts:** Peak-hour warnings and delay severity levels.

<p align="center">
 <img width="1906" height="900" alt="image" src="https://github.com/user-attachments/assets/33ecd809-e7f1-4100-92bc-d13cfb834de1" />
</p>
<p align="center">
 <img width="1880" height="861" alt="image" src="https://github.com/user-attachments/assets/2c6d8bf3-ec79-4314-b44c-c041e287eb2d" />
</p>

-----

## 📄 Research Contributions

This work contributes toward:

- Simulation-driven delay prediction for data-scarce metro systems
- Hybrid operational + machine learning modeling
- Explainable AI applications in smart transportation
- Delay propagation analytics in urban transit systems
- Generalizable transit prediction frameworks

The project bridges the gap between academic ML modeling and practical transportation operations.

---

## 🛠 Tech Stack

### Machine Learning & AI
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- SHAP

### Data Engineering
- Pandas
- NumPy

### Visualization & Analytics
- Matplotlib
- Seaborn

### Dashboard & Deployment
- Streamlit
- Joblib

### Development
- Python
- Jupyter Notebook
- VS Code
- Git & GitHub

---

## 📁 Project Structure

```text
metro-delay-prediction/
│── data/
│   ├── raw/               # Original GTFS files
│   ├── processed/         # Cleaned features
│   └── validation/       # Indian Railways data
│── notebooks/             # EDA & Model Training
│── src/                   # Python scripts for logic
│── models/                # Saved .pkl and .h5 files
│── app.py                 # Streamlit dashboard
│── requirements.txt       # Dependencies
└── README.md
```

-----

## 🚀 Local Setup

1.  **Clone the Repo**

    ```bash
    git clone https://github.com/ahana4banerjee/metro-delay-prediction.git
    cd metro-delay-prediction
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**

    ```bash
    streamlit run src/app.py
    ```

-----

## 🏁 Conclusion

This project successfully combines data engineering, delay simulation, and ensemble learning to create a scalable metro delay prediction framework. It proves that even in the absence of public historical data, reliable predictions can be built using operational insights and smart simulation.

-----

## 👩‍💻 Author

**Ahana Banerjee** 

*3rd Year, ECE IDP, JNTUH*

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/ahana-4-banerjee)
[![Gmail](https://img.shields.io/badge/-Gmail-D14836?style=flat&logo=gmail&logoColor=white)](mailto:banerjeeahana4@gmail.com)
[![Portfolio](https://img.shields.io/badge/-Portfolio-121013?style=flat&logo=vercel&logoColor=white)](https://portfolio-website-seven-roan-36.vercel.app/)

