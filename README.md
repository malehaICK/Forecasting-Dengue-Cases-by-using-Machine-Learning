# Dengue Incidence Forecasting using Machine Learning

## Overview
This project predicts dengue cases using machine learning models trained on meteorological data. It explores how environmental factors influence outbreaks and builds a forecasting pipeline for two cities.

---

## Workflow

```mermaid
flowchart LR
    subgraph Data
        A[Load Dataset]
        B[Preprocessing]
    end

    subgraph Features
        C[Feature Selection]
    end

    subgraph Modeling
        D[Model Training]
    end

    subgraph Evaluation
        E[Evaluation]
        F[Model Comparison]
    end

    A --> B --> C --> D --> E --> F
```

---

## Dataset

<details>
<summary>View dataset details</summary>

### Source
DrivenData DengAI Dataset  

### Features

| Category        | Variables                          |
|----------------|-----------------------------------|
| Climate        | Temperature, Humidity             |
| Weather        | Precipitation                     |
| Environment    | Vegetation Index                  |
| Location       | City-level data                   |

### Cities

| City       | Country        |
|------------|----------------|
| San Juan   | Puerto Rico    |
| Iquitos    | Peru           |

</details>

---

## Methodology

```mermaid
flowchart LR
    A[Raw Dataset] --> B[Data Preprocessing]
    B --> C[Feature Selection]
    C --> D[Model Training]
    D --> E[Evaluation]
    E --> F[Model Comparison]
```

<details>
<summary>View detailed steps</summary>

### Data Preprocessing
- Missing values handled using mean imputation  
- Temperature unit conversion  
- Rounding for consistency  

### Feature Selection
- Correlation analysis to identify important features  
- Removed negatively correlated and weak features  

### Models Used
- K-Nearest Neighbors (KNN)  
- Random Forest (RF)  
- Gradient Boosting Regressor (GBR)  
- Support Vector Regressor (SVR)  

### Evaluation Metric
- Mean Absolute Error (MAE)  

</details>

---

## Results

### Model Performance (MAE ↓)

| Model | San Juan | Iquitos | Performance |
|------|----------|---------|-------------|
| KNN  | Lowest   | Low     | Best overall |
| RF   | Medium   | Medium  | Stable |
| GBR  | Medium   | Medium  | Comparable |
| SVR  | Low      | Lowest  | Best for Iquitos |

---

### Impact of Feature Selection

```mermaid
flowchart LR
    A[All Features] --> B[Correlation Analysis]
    B --> C[Selected Features]
    C --> D[Improved Accuracy]
```

- San Juan: 16 → 11 features  
- Iquitos: 16 → 8 features  
- Reduced noise and improved model performance  

---

### Key Findings

- KNN achieved the best overall performance  
- SVR performed best for Iquitos  
- Feature selection significantly improved results  
- Weather variables strongly influence dengue cases  

---

### Final Insight

Lower MAE indicates better prediction accuracy. KNN proved to be the most reliable model for dengue forecasting in this study.

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

---

## How to Run

```
git clone <your-repo-link>
cd <project-folder>
pip install -r requirements.txt
python main.py
```

---

## Future Work

- Apply deep learning models  
- Incorporate additional environmental and demographic data  
- Use time-series models such as ARIMA  
- Deploy as a real-time forecasting system  
