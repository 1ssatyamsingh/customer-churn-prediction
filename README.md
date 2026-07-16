# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn based on customer demographics, service subscriptions, contract details, and billing information.

The project covers the complete machine learning workflow, including data preprocessing, exploratory data analysis, class imbalance handling, model comparison, evaluation, and a reusable prediction system.

---

## Project Overview

Customer churn prediction helps businesses identify customers who are likely to discontinue their services. This project uses the **Telco Customer Churn dataset** to build a binary classification model that predicts:

* `1` → Customer will churn
* `0` → Customer will not churn

The project compares multiple machine learning models and selects the best-performing model based on cross-validation performance.

---

## Features

* Data loading and dataset inspection
* Data cleaning and preprocessing
* Handling missing values in `TotalCharges`
* Exploratory Data Analysis (EDA)
* Numerical feature distribution analysis
* Outlier analysis using box plots
* Correlation analysis
* Categorical feature analysis
* Label encoding of categorical features
* Class imbalance handling using SMOTE
* Comparison of multiple classification algorithms
* 5-fold cross-validation
* Model evaluation using:

  * Accuracy
  * Confusion Matrix
  * Precision
  * Recall
  * F1-score
* Model and encoder serialization using Pickle
* Prediction system for new customer data

---

## Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Loading & Inspection
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Encoding
     ↓
Train-Test Split
     ↓
SMOTE Oversampling
     ↓
Model Training
     ↓
5-Fold Cross-Validation
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
Prediction on New Customer Data
```

---

## Dataset

The project uses the **Telco Customer Churn Dataset**.

The dataset contains information about:

### Customer Information

* Gender
* Senior Citizen status
* Partner
* Dependents

### Service Information

* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

### Account Information

* Tenure
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

### Target Variable

```text
Churn
```

* `Yes` → Customer churned
* `No` → Customer did not churn

---

## Data Preprocessing

### 1. Removing Unnecessary Features

The `customerID` column was removed because it is a unique identifier and does not provide meaningful predictive information.

```python
df = df.drop(columns=["customerID"])
```

### 2. Handling Missing Values

Some records contained blank values in the `TotalCharges` column. These values were replaced with `0.0` and converted to a numerical data type.

```python
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)
```

### 3. Target Encoding

The target variable was converted into binary values:

```text
Yes → 1
No  → 0
```

```python
df["Churn"] = df["Churn"].replace({
    "Yes": 1,
    "No": 0
})
```

### 4. Categorical Feature Encoding

Categorical features were converted into numerical values using `LabelEncoder`.

The encoders were saved using Pickle so that the same transformations could be applied to new input data during prediction.

```python
encoders = {}

for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder
```

---

## Exploratory Data Analysis

The project performs exploratory analysis to understand:

* Numerical feature distributions
* Mean and median values
* Potential outliers
* Correlations between numerical features
* Distribution of categorical features
* Class distribution of the target variable

### Visualizations Used

* Histograms with KDE
* Box plots
* Correlation heatmaps
* Count plots

---

## Handling Class Imbalance

The dataset contains an imbalance between customers who churned and customers who did not churn.

To address this, **Synthetic Minority Oversampling Technique (SMOTE)** was applied only to the training data.

```python
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
```

Applying SMOTE only to the training set helps prevent data leakage into the test set.

---

## Machine Learning Models

The following classification models were trained and compared:

### Decision Tree

```python
DecisionTreeClassifier(
    random_state=42
)
```

### Random Forest

```python
RandomForestClassifier(
    random_state=42
)
```

### XGBoost

```python
XGBClassifier(
    random_state=42
)
```

---

## Model Comparison

Each model was evaluated using **5-fold cross-validation**.

```python
scores = cross_val_score(
    model,
    X_train_smote,
    y_train_smote,
    cv=5,
    scoring="accuracy"
)
```

The average cross-validation accuracy was calculated for each model.

Based on the cross-validation results, the **Random Forest classifier** achieved the best performance among the evaluated models and was selected as the final model.

---

## Model Evaluation

The final Random Forest model was evaluated on the original unseen test dataset using:

### Accuracy

Measures the overall percentage of correct predictions.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

```python
print(accuracy_score(y_test, y_test_pred))

print(confusion_matrix(
    y_test,
    y_test_pred
))

print(classification_report(
    y_test,
    y_test_pred
))
```

---

## Model Serialization

The trained model and feature names were saved using Pickle.

```python
model_data = {
    "model": rfc,
    "features_names": X.columns.tolist()
}

with open(
    "customer_churn_model.pkl",
    "wb"
) as f:
    pickle.dump(model_data, f)
```

The categorical encoders were also saved:

```python
with open(
    "encoders.pkl",
    "wb"
) as f:
    pickle.dump(encoders, f)
```

This allows the model and preprocessing logic to be reused for future predictions.

---

## Prediction System

The project supports prediction on new customer data.

Example input:

```python
input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}
```

The saved encoders are loaded and applied to categorical features before making predictions.

```python
prediction = loaded_model.predict(input_data_df)

prediction_probability = loaded_model.predict_proba(
    input_data_df
)
```

Example output:

```text
Prediction: Churn
Prediction Probability: [...]
```

---

## Project Structure

```text
customer-churn-prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_prediction.ipynb
├── customer_churn_model.pkl
├── encoders.pkl
├── requirements.txt
└── README.md
```

---

## Technologies Used

### Programming Language

* Python

### Data Analysis

* NumPy
* Pandas

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn

### Model Persistence

* Pickle

---

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd customer-churn-prediction
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Create a `requirements.txt` file containing:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

---

## Running the Project

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
customer_churn_prediction.ipynb
```

Run the notebook cells sequentially to:

1. Load the dataset
2. Perform data preprocessing
3. Analyze the data
4. Train the models
5. Compare model performance
6. Evaluate the final model
7. Save the model and encoders
8. Generate predictions for new customers

---

## Key Machine Learning Concepts Demonstrated

* Supervised Learning
* Binary Classification
* Exploratory Data Analysis
* Feature Preprocessing
* Label Encoding
* Class Imbalance
* SMOTE
* Cross-Validation
* Ensemble Learning
* Random Forest
* XGBoost
* Model Evaluation
* Model Serialization
* Predictive Systems

---

## Future Improvements

* Add hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
* Evaluate models using ROC-AUC in addition to accuracy
* Add feature importance visualization
* Build an interactive prediction interface using Streamlit
* Deploy the model as a REST API using FastAPI
* Use a preprocessing pipeline to combine transformations and model inference
* Add probability threshold tuning to optimize recall for churn detection

---

## Author

**Satyam Singh**

Computer Science and Engineering Student
Interested in Machine Learning, Generative AI, and Software Development.
