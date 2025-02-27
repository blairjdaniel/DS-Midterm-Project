# 🏡 House Price Prediction Model

## 📌 Project Overview
This project aims to predict house sale prices using machine learning techniques. We explore real estate data, preprocess it, engineer features, and build predictive models to achieve the best accuracy.

## 🎯 Goals
- Clean and preprocess raw housing data.
- Engineer features to enhance model performance.
- Apply machine learning models to predict sold prices.
- Evaluate and refine model performance.

---

## 🔄 Process

### 🛠 Data Preprocessing & Feature Engineering
1. **Data Loading & Cleaning**
   - Merged multiple `.json` files into a single DataFrame.
   - Handled missing values using KNN imputation and mean imputation.
   - Removed outliers based on the 5th and 95th percentiles of `sold_price`.
   - Dropped irrelevant or highly correlated features.

2. **Feature Engineering**
   - Created new features such as `price_per_sqft` and `median_value_per_sqft`.
   - Encoded categorical variables using Label Encoding.
   - Normalized and standardized features for improved model performance.
   - Applied Principal Component Analysis (PCA) to reduce dimensionality.

3. **Exploratory Data Analysis (EDA)**
   - Identified key trends and correlations using visualizations.
   - Scatterplot of features vs. `sold_price`:
     ![Sold Price Scatterplot](images/Sold%20Price%20scatterplot%20correlation.png)
   - Boxplot analysis to detect outliers:
     ![Sold Price Boxplot](images/Sold%20Price%20BoxPlot%20Correlation.png)
   - Feature importance visualization:
     ![Feature Importance](images/Feature%20Importance.png)

---

## 🤖 Model Selection & Evaluation
We experimented with different datasets and transformations, including:
- **Unprocessed data**
- **Polynomial features**
- **PCA-transformed data**
- **Polynomial + Scaled + PCA data**

### ✅ Best Performing Model: **Gradient Boosting Regressor**
- **Best Hyperparameters:**  
  `{ 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 300, 'subsample': 0.8 }`
- **Performance Metrics:**
  - **Mean Squared Error (MSE):** 6,399,959,654.03
  - **Root Mean Squared Error (RMSE):** 79,999.75
  - **Mean Absolute Error (MAE):** 55,246.17
  - **R² Score:** 0.816
  - **Adjusted R² Score:** 0.8039

### 🔥 Other Models Tested
#### XGBoost:
- **MSE:** 6,466,147,328.0
- **RMSE:** 80,412.36
- **MAE:** 57,161.27
- **R² Score:** 0.8141
- **Adjusted R² Score:** 0.8019

#### Random Forest:
- **Best Hyperparameters:**  
  `{ 'bootstrap': True, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200 }`
- **Performance Metrics:**
  - **Mean Absolute Error (MAE):** 64,223.21
  - **R² Score:** 0.8089
  - **Adjusted R² Score:** 0.7963

---

## 🚧 Challenges
- **Handling missing values:** Certain attributes had too many missing values and required imputation.
- **High-dimensional data:** Reducing unnecessary features was crucial to improving performance.
- **Outliers:** Needed careful filtering of extreme price values.

---

## 🚀 Future Improvements
- **Hyperparameter tuning:** Further optimizing Gradient Boosting/XGBoost for better performance.
- **Feature selection:** Experimenting with more domain-specific features to enhance prediction accuracy.
- **Ensemble models:** Combining multiple models to improve overall performance.

---

📌 **Conclusion:** The Gradient Boosting model provided the best predictive accuracy, but there is room for further improvement through advanced hyperparameter tuning and feature selection.
