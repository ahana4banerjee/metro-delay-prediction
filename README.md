# Metro Train Delay Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
This project presents a **machine learning-driven framework** for **predicting metro train delays** using structured transit schedule data enriched with operational dynamics.

To ensure robustness and real-world relevance, the developed approach was validated on an independent **Indian Railways delay dataset**, where it demonstrated **consistent performance metrics and behavioral patterns**, confirming strong generalization across large-scale rail networks.

---

## Objective
The goal is to accurately predict train delay (in minutes) using key transit and operational features:

- Station information
- Time of day
- Peak vs non-peak hours
- Stop sequence (trip progression)
- Route direction
- Interchange congestion impact
- Delay propagation

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Folium (for map visualization)
- Joblib (model persistence)
- Streamlit (for dashboard)

---

## Project Structure

```text
metro-delay-prediction/
│
├── data/
│   ├── raw/   # Static Hyderabad GTFS Dataset
│   ├── processed/   # Processed dataset for ML
|   ├── indian_rail_delay/  # Indian Railways Delay Dataset for validation 
│
├── notebooks/
    ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   ├── advanced_preprocessing.ipynb
|   ├── validation_indian_rail.ipynb
│
├── src/
|   ├── config.py
│   ├── utils.py
│   ├── app.py

│
├── models/
│
├── reports/
|   ├── figures
│   ├── validation_results
|   ├── cross_dataset_analysis
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Methodology

### Data Processing & Enrichment

- GTFS schedule ingestion and normalization
- Temporal and route-based structuring
- Incorporation of operational delay dynamics

### Feature Engineering

- Time-based features (hour, peak indicators)
- Route progression (stop index)
- Station importance and interchange encoding
- Delay propagation modeling

### Model Development

- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- Custom Weighted Delay Model

## Workflow
1. Data Collection
2. Data Cleaning
3. Delay Simulation
4. Feature Engineering
5. Model Training
   - Linear Regression
   - Decision Tree
   - Random Forest
   - XGBoost 
   - Custom Weighted Delay Model
7. Evaluation Metrics
   - MAE
   - RMSE
   - R² Score
8. Visualization
9. Dashboard Integration using Streamlit

---

## Observations & Insights (Metro Data)

- Peak hours significantly increase delays
- Interchange stations show consistent congestion
- Delay propagation is a dominant factor
- Tree-based models outperform linear models
- Custom model is interpretable but less accurate

---

## Cross Dataset Validation using Indian Railways Dataset

The approach was rigorously validated on an independent Indian Railways delay dataset to assess real-world applicability.

### Observations and Inference

- Comparable performance metrics (MAE, RMSE, R² trends) across datasets
- Strong consistency in peak-hour delay behavior
- Similar delay propagation patterns observed
- Major stations/junctions exhibit higher delay impact
- Right-skewed delay distribution replicated
- Tree-based models consistently outperform linear models

### Conclusion

The alignment in both quantitative performance metrics and qualitative behavioral patterns demonstrates that the model captures fundamental delay dynamics, making it robust, transferable, and reliable across different rail systems.

---

## Dashboard

An interactive interface is provided for real-time predictions.

### Inputs from User

- Source station
- Destination station
- Time of travel
- Day/peak hour selection

### Processing

- Input is converted into model features
- Same preprocessing steps are applied
- Trained model is loaded using Joblib
- Delay prediction is generated
  
### Outputs Shown

- Predicted delay (in minutes)
- Peak vs non-peak indication using colour codings
- Total stops and estimated duration
- Insights on peak hour, traffic, and delay

---

## Local Setup & Execution

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd metro-delay-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebooks

#### Option A (Recommended - Manual)

- Open notebooks folder
- Run all notebooks in order:
  - data_exploration.ipynb
  - advanced_preprocessing.ipynb
  - model_building.ipynb
  - validation_indian_rail.ipynb
- Click “Run All Cells” in each notebook

#### Option B (Shortcut - CLI)

```bash
jupyter nbconvert --to notebook --execute notebooks/*.ipynb
```

### 4. Observe Outputs

After running notebooks, you will get:

- Graphs & visualizations
- Delay patterns
- Model performance metrics (MAE, RMSE, R²)
- Summary insights

### 5. Launch UI Dashboard

```bash
streamlit run src/app.py
```

### 6. Interact with Dashboard

- Enter station, time, etc.
- View predicted delay
- Explore visual insights

---

## Future Work

- Real-time data
- Time-series and LSTM-based modeling
- Advanced interactive dashboard
- API deployment

---

### Final Note

This project demonstrates a production-oriented ML pipeline with strong emphasis on generalization, validation, and system-level understanding of delay behavior, validated across multiple railway networks.



    
