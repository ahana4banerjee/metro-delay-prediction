# Metro Train Delay Prediction

## Overview
This project focuses on predicting metro train delays using machine learning. It leverages GTFS (General Transit Feed Specification) schedule data and a realistic delay simulation strategy to approximate real-world conditions.

To ensure credibility, the model behavior was also validated on an Indian Railways delay dataset, confirming that learned patterns generalize beyond simulated data.

---

## Objective
To predict train delay (in minutes) using:

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
- Scikit-learn
- Matplotlib, Seaborn
- Folium (for map visualization)
- Joblib (model persistence)

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
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

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

## Validation on Indian Railways Dataset

To verify realism, the approach was tested on an Indian Railways delay dataset.

### Similar Patterns Observed
1. Peak hour effect exists
→ Higher delays during busy hours

2. Delay propagation is strong
→ Late trains continue to remain late

3. Station importance matters
→ Major junctions show higher delays

4. Non-linear behavior dominates
→ Tree-based models again outperform linear models

5. Delay distribution similarity
→ Right-skewed distribution (many small delays, few large delays)

**_The simulated metro dataset successfully captures real-world delay behavior, making the model reliable despite lack of real-time metro data._**

---

## Dashboard

The project includes a simple UI (planned or implemented using Streamlit/Folium) to interact with predictions.

### Inputs from User

- Source station
- Destination station
- Time of travel
- Day / peak hour selection

### What Happens Internally

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

### 5. Run UI Dashboard

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
- LSTM models
- Enhance UI dashboard
- API deployment



    
