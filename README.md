# 🏠 London House Price Prediction - Advanced Techniques 🏠

Welcome to the **London House Price Prediction - Advanced Techniques** project! This repository contains a solution for the [Kaggle competition](https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques/overview) focused on predicting house prices in London using advanced machine learning techniques. The goal is to predict the `price` of properties in the test dataset based on features like location, size, and property characteristics. 📊

## 📋 Project Overview

This project is part of the Kaggle competition *London House Price Prediction - Advanced Techniques*. The objective is to build a machine learning model to predict house prices in London using a dataset with property details such as bedrooms, bathrooms, floor area, tenure, and more. The solution in this repository uses **XGBoost** and **SGDRegressor** models, with hyperparameter tuning via GridSearchCV, to generate predictions for the test set.

### 🎯 Objective
- Predict house prices (`price`) for the test dataset.
- Handle missing data and preprocess features effectively.
- Submit predictions to Kaggle in the required format to compete on the leaderboard.

## 📂 Dataset

The dataset is provided by the [Kaggle competition](https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques/data) and includes:
- **train.csv**: Training data with property details and the target variable `price`.
- **test.csv**: Test data with property details (no `price` column) for predictions.
- **sample_submission.csv**: Template for the submission file format.

Key features in the dataset:
- `fullAddress`, `postcode`, `outcode`: Location details.
- `latitude`, `longitude`: Geographic coordinates.
- `bedrooms`, `bathrooms`, `floorAreaSqM`, `livingRooms`: Property characteristics.
- `tenure`, `propertyType`, `currentEnergyRating`: Property attributes.
- `sale_month`, `sale_year`: Transaction details.

## 🛠️ Installation

To run the code, ensure you have the following dependencies installed:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost
```

Clone the repository to your local machine:

```bash
git clone https://github.com/SamvelStepanyan4/London_House_prediction.git
cd London_House_prediction
```

## 🚀 Usage

1. **Download the Dataset**: Download `train.csv`, `test.csv`, and `sample_submission.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques/data).
2. **Place Files**: Store the dataset files in the same directory as the notebook (`London_House.ipynb`).
3. **Run the Notebook**:
   - Open `London_House.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells sequentially to preprocess the data, train the model, and generate predictions.
4. **Output**: The notebook generates a `sub.csv` file with predicted house prices for submission to Kaggle.

## 🔍 Methodology

### 1. Data Preprocessing
- **Missing Values**: Missing data in columns like `bathrooms` (18.2%), `bedrooms` (9.3%), `floorAreaSqM` (5.2%), `livingRooms` (13.9%), `tenure` (2.1%), and `currentEnergyRating` (21.3%) are handled using a custom `fill_na` function. This function imputes missing values by randomly sampling from the column's unique values based on their probability distribution.
- **Feature Dropping**: The `ID` column is removed as it’s irrelevant for prediction.
- **Data Splitting**: The training data is split into features (`X_train`) and target (`y_train`), with `X_test` used for predictions.

### 2. Model Training
- **SGDRegressor**:
  - Hyperparameter tuning using GridSearchCV with parameters: `alpha` ([0.001, 0.01, 0.1, 1]), `eta0` ([0.001, 0.01, 0.1, 1]), `max_iter` ([100, 500, 1000]), `penalty` (['l2', 'l1', 'elasticnet', None]), and `early_stopping` ([True, False]).
  - Best parameters: `{'alpha': 0.01, 'early_stopping': False, 'eta0': 1, 'max_iter': 100, 'penalty': 'l1'}`.
- **XGBoost**:
  - Initial model with minimal hyperparameter tuning (`n_estimators=[1]`) using GridSearchCV.
  - Note: The current setup uses a single estimator for simplicity, but performance can be improved with further tuning.

### 3. Prediction and Submission
- Predictions are generated for the test set using the trained model.
- The results are saved to `sub.csv` in the format required by Kaggle (matching the `sample_submission.csv` structure).
- The submission file can be uploaded to the [Kaggle competition](https://www.kaggle.com/competitions/london-house-price-prediction-advanced-techniques/submit) for evaluation.

## 📈 Results

The notebook produces a submission file (`sub.csv`) with predicted house prices. The performance is evaluated on Kaggle’s leaderboard using the competition’s evaluation metric (likely **Root Mean Squared Error (RMSE)**, though not explicitly stated in the notebook). To improve results, consider:
- Expanding the XGBoost hyperparameter grid (e.g., `learning_rate`, `max_depth`, `subsample`).
- Adding feature engineering, such as encoding `postcode` or calculating distance to central London.
- Testing ensemble methods or other models like LightGBM or CatBoost.

## 📝 Future Improvements
- 🧹 **Feature Engineering**: Create features like distance to city center, postcode-based clustering, or property age.
- 🔧 **Hyperparameter Tuning**: Expand the XGBoost parameter grid (e.g., `n_estimators=[50, 100, 200]`, `max_depth=[3, 5, 7]`, `learning_rate=[0.01, 0.1, 0.3]`).
- 📊 **Model Evaluation**: Implement local cross-validation to compute RMSE or other metrics before submission.
- 🤖 **Advanced Models**: Experiment with ensemble methods or gradient boosting libraries like LightGBM or CatBoost.
- 🗺️ **Geospatial Analysis**: Leverage `latitude` and `longitude` for spatial features or visualizations.

## 🙌 Contributing

Contributions are welcome! To contribute:
- Open an issue to report bugs or suggest improvements.
- Submit a pull request with enhancements to preprocessing, modeling, or documentation.

Please follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

## 🌟 Acknowledgments
- Thanks to [Kaggle](https://www.kaggle.com) for hosting the *London House Price Prediction - Advanced Techniques* competition.
- Built with ❤️ using Python, pandas, scikit-learn, and XGBoost.