# Metro Train Delay Prediction

## Overview
This project focuses on predicting metro train delays using machine learning.
It leverages GTFS (General Transit Feed Specification) schedule data and simulates realistic delay patterns to build and evaluate predictive models.

## Objective
To build a machine learning model that predicts train delay (in minutes) using features such as:

- Station information
- Time of day
- Peak vs non-peak hours
- Trip progression (stop sequence)
- Route direction
- Interchange station impact

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Folium (for map visualization)
- Joblib (model persistence)

## Project Structure

```text
metro-delay-prediction/
│
├── data/
│   ├── raw/   # Static Hyderabad GTFS Dataset
│   └── processed/   # Processed dataset for ML
│
├── notebooks/
    ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   ├── advanced_preprocessing.ipynb
│
├── src/
│
├── models/
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```


## Workflow
1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Model Training
5. Evaluation (RMSE, MAE)
6. Visualization

## Models to be Used
- Linear Regression
- Random Forest

## Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## Timeline
- Phase 1: Data collection & EDA  
- Phase 2: Preprocessing & features  
- Phase 3: Modeling  
- Phase 4: Visualization  
- Phase 5: Final report

## Current Status

- ✅ Phase 1 Completed
- ✅ Phase 2 Completed
- ⏳ Phase 3 In Progress


## Future Work
- Real-time prediction
- Dashboard integration
