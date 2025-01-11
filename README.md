# Titanic Survival Prediction using XGBoost with Hyperparameter Optimization

This project predicts the survival of Titanic passengers using the XGBoost algorithm. It includes data preprocessing, feature engineering, and hyperparameter optimization using randomized search. The project is implemented in Python.

## Project Structure

- **Data Preprocessing**:
  - Handles missing values in the `Age`, `Fare`, and `Embarked` columns.
  - Creates categorical bins for `Age` and `Fare`.
  - Generates dummy variables for categorical features like `Sex`, `Pclass`, and `Embarked`.
  - Extracts features from `Ticket` and `Cabin` columns.
  - Removes irrelevant columns like `Name`, `Ticket`, and `Cabin`.

- **Model Training and Optimization**:
  - Splits the training data into training and validation sets.
  - Uses `RandomizedSearchCV` to optimize hyperparameters for the XGBoost model.
  - Evaluates the model on both training and validation datasets.

- **Output**:
  - Generates predictions for the test dataset.
  - Creates a submission file (`submission.csv`) for evaluation.

## Key Features

- **Hyperparameter Tuning**:
  - Optimizes `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree` for the XGBoost model.
  - Employs `RandomizedSearchCV` for efficient hyperparameter exploration.

- **Performance Metrics**:
  - Calculates accuracy on both training and validation datasets.

## Dependencies

- Python 3.8 or higher
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `scipy`

Install dependencies using pip:

```bash
pip install pandas numpy scikit-learn xgboost scipy
