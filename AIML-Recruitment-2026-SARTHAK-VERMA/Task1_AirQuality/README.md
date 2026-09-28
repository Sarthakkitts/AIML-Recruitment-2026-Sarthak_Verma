## Overview
Forecasts NO2 levels one hour ahead using UCI Air Quality dataset.

## Dataset
- Source: AirQualityUCI.csv (hourly averaged sensor data)
- Target: NO2 levels (next hour prediction)

## Methodology
1. **Preprocessing**: Handle -200 missing values, drop >30% missing columns, ffill()+bfill()
2. **Features**: Time features (hour, day, month), cyclical encoding, lagged pollutants, rolling statistics
3. **Models**: Linear Regression vs Random Forest (chronological 80/20 split)
4. **Evaluation**: MAE, MSE, RMSE, R²

## Results
- Best Model: [Check notebook output]
- R² Score: [Check notebook output]
- Key Insights: Feature importance if Random Forest performed best

## Files
- `notebooks/air_quality_forecasting.ipynb`: Complete analysis
- `data/AirQualityUCI.csv`: Dataset
- `README.md`: This file

## Requirements
- PPIP install pandas numpy matplotlib seaborn scikit-learn tensorflow
EOF
