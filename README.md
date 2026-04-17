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

## ⚙️ Core Methodology

### 🧠 Delay Simulation Logic

Since real Hyderabad Metro delay data is not publicly available, delays were simulated using a multi-factor formula:
$$Delay = Base + Peak + Congestion + Propagation + Noise$$

Key factors included:

  * **Peak-hour congestion:** Higher weight during morning/evening rushes.
  * **Interchange stations:** Increased dwell time at busy hubs.
  * **Propagation:** Delay from the previous station carried forward.

### 🏗 Feature Engineering

  * **Temporal:** Hour of day, Day of week, Peak/Non-peak flags.
  * **Spatial:** Station encoding, Route encoding, Stop sequence.
  * **Sequential:** Previous station delay, delay propagation component.
  * **Operational:** Interaction terms (Peak × Congestion).

-----

## 🤖 Model Performance

### 📊 Metro Dataset Results

| Model | RMSE | MAE | $R^2$ |
| :--- | :--- | :--- | :--- |
| **XGBoost** | **1.144** | **0.829** | **0.940** |
| Random Forest | 1.159 | 0.841 | 0.938 |
| Linear Regression | 1.325 | 0.976 | 0.913 |
| LSTM | 1.352 | 1.025 | 0.920 |

### 🚆 Indian Railways Validation

| Model | RMSE | MAE | $R^2$ |
| :--- | :--- | :--- | :--- |
| Random Forest | 7.16 | 3.21 | 0.97 |
| **XGBoost** | **8.13** | **2.97** | **0.96** |
| LSTM | 13.86 | 6.86 | 0.88 |
| Linear Regression | 15.66 | 7.81 | 0.85 |

**Key Insight:** Tree-based ensemble models (XGBoost/Random Forest) maintained the strongest performance and generalization across both datasets.

-----

## 🔍 Key Findings

1.  **Previous Delay:** The strongest predictor of future delay (propagation effect).
2.  **Peak Hours:** Significantly increase delay variance.
3.  **Busy Hubs:** Interchange stations show higher delay tendencies.
4.  **Generalization:** Models trained on simulated logic successfully identified patterns in real Indian Railways data.

-----

## 💻 Dashboard Features

### 🌐 Try the Live App
https://metro-delay-prediction.streamlit.app/

Built using **Streamlit**, the dashboard provides:

  * **User Inputs:** Source station, Destination, Day, and Time.
  * **Real-time Predictions:** Predicted delay and estimated arrival time.
  * **Operational Alerts:** Peak-hour warnings and delay severity levels.

<p align="center">
  <img src="https://github.com/user-attachments/assets/613a75a6-688e-4a1c-8c12-33de875159bd" width="48%" />
  <img src="https://github.com/user-attachments/assets/49bfca37-421e-4140-a8f0-8ffe216097ac" width="48%" />
</p>

-----

## 🛠 Tech Stack

  * **Language:** Python
  * **Data Science:** Pandas, NumPy, Scikit-learn
  * **ML Models:** XGBoost, TensorFlow (LSTM)
  * **Visualization:** Matplotlib, Seaborn
  * **Deployment:** Streamlit
  * **Serialization:** Joblib

-----

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

