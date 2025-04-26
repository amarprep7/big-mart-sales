# Big Mart Sales Prediction: Approach Note

## Overview
This project aims to predict sales for Big Mart outlets using historical sales data and various item/outlet features. The process follows a modular, reproducible machine learning pipeline, with clear separation of data processing, feature engineering, model training, and inference.

## Thought Process & Experimentation Steps

### 1. Project Structure & Modularity
- Adopted a standard ML project structure with separate folders for data, models, logs, notebooks, source code, and documentation.
- Modularized the code into logical components: configuration, logging, data preprocessing, training, and prediction.

### 2. Data Preparation
- Raw data was placed in `data/raw/` and processed data in `data/processed/`.
- Preprocessing included:
  - Handling missing values (e.g., interpolating weights, filling outlet sizes based on type).
  - Standardizing categorical values (e.g., Item_Fat_Content).
  - Feature extraction (e.g., extracting item type, calculating outlet age).
  - Feature engineering (e.g., price per weight, store age-size interaction, visibility-MRP interaction).
  - Ordinal encoding for categorical variables.

### 3. Experiments for Best Results
- **Baseline Models:** Started with simple regressors (Linear Regression, Random Forest) to establish a baseline.
- **Feature Engineering:** Created new features (e.g., Price_per_Weight, Store_Age_Size, Visibility_MRP) and tested their impact on model performance.
- **Feature Selection:** Used XGBoost feature importance to select the most predictive features. Multiple runs were done with different numbers of top features (6, 10, 20) to find the optimal set.
- **Hyperparameter Tuning:** Performed RandomizedSearchCV on XGBoost with a wide range of parameters (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda). The best parameters were selected based on cross-validation R2 score.
- **Cross-Validation:** Used 5-fold cross-validation to ensure model robustness and avoid overfitting.
- **Model Comparison:** Compared XGBoost with Random Forest and found XGBoost consistently outperformed in terms of R2 and generalization.
- **Experiment Tracking:** All experiments, including parameter grids and results, were logged for reproducibility.

### 4. Final Model Training & Inference
- The final XGBoost model was trained on the full training set using the best hyperparameters and top features.
- The exact feature list was saved and used during inference to ensure consistency.
- Predictions were generated for the test set and saved in a submission-ready CSV file.

### 5. Logging & Reproducibility
- All major steps and results are logged to `logs/pipeline.log` for traceability.
- Experimentation scripts and notebooks are kept in the `notebooks/` folder for iterative development and analysis.
- Both the model and the feature list are saved for reproducibility and reliable deployment.

## FastAPI Endpoint Note

A FastAPI endpoint is provided in `src/main.py` to serve the trained model for real-time inference. The `/predict` endpoint accepts a POST request with a list of data rows (in JSON format), preprocesses the input using the same pipeline as training, and returns predictions. This enables easy integration of the model into applications or dashboards.

- **Request Example:**
  ```json
  {
    "data": [
      {
        "Item_Identifier": "FD",
        "Item_Weight": 9.3,
        "Item_Fat_Content": "Low Fat",
        "Item_Visibility": 0.016047301,
        "Item_Type": "Dairy",
        "Item_MRP": 249.8092,
        "Outlet_Identifier": "OUT049",
        "Outlet_Establishment_Year": 1999,
        "Outlet_Size": "Medium",
        "Outlet_Location_Type": "Tier 1",
        "Outlet_Type": "Supermarket Type1"
      },
      {
        "Item_Identifier": "DR",
        "Item_Weight": 5.92,
        "Item_Fat_Content": "Regular",
        "Item_Visibility": 0.019278216,
        "Item_Type": "Soft Drinks",
        "Item_MRP": 48.2692,
        "Outlet_Identifier": "OUT018",
        "Outlet_Establishment_Year": 2009,
        "Outlet_Size": "Medium",
        "Outlet_Location_Type": "Tier 3",
        "Outlet_Type": "Supermarket Type2"
      }
    ]
  }
  ```
- **Response Example:**
  ```json
  {
    "predictions": [
      1800.5,
      2100.7
    ]
  }
  ```

This API ensures the same preprocessing and feature selection as the training pipeline, providing consistent and reliable predictions.

## Key Learnings
- Systematic experimentation with feature engineering and hyperparameter tuning is crucial for achieving the best results.
- Consistent feature usage between training and inference prevents errors and ensures reliable predictions.
- Modular code and clear logging greatly simplify debugging and future enhancements.

