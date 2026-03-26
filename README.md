# 🚕 Uber Fare Amount Prediction Using Regression Analysis

A Machine Learning project that predicts the fare amount of future 
Uber rides based on ride features like distance, time, and passenger count.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `Fare Amount Of Future Rides - comp2.ipynb` | Data loading, cleaning & feature engineering |
| `Uber-prediction-final.ipynb` | Model training, tuning & final results |

---

## 📌 Project Workflow

### 1. Data Understanding
- Dataset: 200,000 Uber ride records
- Features: pickup datetime, coordinates, passenger count, fare amount

### 2. Data Preprocessing
- Handled missing values (dropped 1 null row)
- Removed negative fare values
- Extracted datetime features: day, hour, month, year

### 3. Feature Engineering
- Extracted **pickup_day, pickup_hour, pickup_month, pickup_year** from datetime
- Calculated **ride distance (in km)** using the Haversine formula from coordinates

### 4. Outlier Treatment
- **Distance** → Capped at 99th percentile
- **Fare Amount** → Floored at 1st percentile, Capped at 99th percentile  
- **Passenger Count** → Filtered to valid range (1–6)

### 5. Correlation Analysis
| Feature | Correlation with Fare |
|---|---|
| distance | **0.85** ✅ |
| pickup_year | 0.10 |
| pickup_month | 0.03 |
| passenger_count | 0.01 |
| pickup_day | 0.005 |
| pickup_hour | -0.02 |

### 6. Model Training & Comparison

| Model | MSE | R² Score |
|---|---|---|
| Decision Tree (max_depth=8) | 0.0532 | **0.782** 🥇 |
| Linear Regression - All Features | 0.0787 | 0.678 |
| Linear Regression - Distance Only | 0.0837 | 0.657 |
| Decision Tree - No Depth Limit | 0.1067 | 0.563 |

### 7. Hyperparameter Tuning
- Plotted Train R² vs Test R² across depths 2–19
- Identified **max_depth = 8** as the optimal value
- Prevented overfitting while maximizing generalization

---

## 🔍 Key Findings
- **Distance** is the strongest predictor with **0.85 correlation** with fare
- Untuned Decision Tree severely overfits (Train R²: 0.997 vs Test R²: 0.563)
- Tuning max_depth from unlimited → 8 improved Test R² from **0.563 → 0.782**
- All features outperform distance-only for both models

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-lightblue)

- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Models**: Linear Regression, Decision Tree Regressor
- **Concepts**: Haversine Formula, MinMax Scaling, Outlier Treatment, 
  Hyperparameter Tuning, Bias-Variance Tradeoff

---

## 👩‍💻 Author
**Nikita** — [GitHub Profile](https://github.com/nikitaj98-prog)
```

