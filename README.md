# Dengue Incidence Forecasting using Machine Learning

## Overview
This project focuses on predicting dengue cases using machine learning models based on meteorological data. The goal is to understand how weather conditions influence dengue outbreaks and to build an effective forecasting model.

## Dataset
- Source: DrivenData DengAI dataset
- Features include:
  - Temperature
  - Humidity
  - Precipitation
  - Vegetation Index
  - Location data
- Two cities analyzed:
  - San Juan
  - Iquitos

## Methodology
1. Data preprocessing:
   - Handled missing values using mean imputation
   - Converted temperature units
   - Rounded values for consistency
2. Feature selection:
   - Used correlation analysis to select important features
3. Model training:
   - K-Nearest Neighbors (KNN)
   - Random Forest (RF)
   - Gradient Boosting Regressor (GBR)
   - Support Vector Regressor (SVR)
4. Evaluation:
   - Performance measured using Mean Absolute Error (MAE)

## Results
- KNN performed best overall for dengue prediction
- Feature selection improved model accuracy
- Different models performed better depending on the city:
  - San Juan: KNN performed best
  - Iquitos: SVR performed well

## Key Insights
- Meteorological factors have a strong impact on dengue outbreaks
- Feature selection significantly improves model performance
- Machine learning can be effectively used for disease forecasting

## Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## How to Run
1. Clone the repository
2. Install dependencies
3. Run the main script or notebook

## Future Work
- Use more advanced models (e.g., deep learning)
- Include additional environmental and demographic data
- Deploy as a real-time prediction system

