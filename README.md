Customer Churn Prediction Pipeline

## Objective
The objective of this project is to develop a robust machine learning pipeline to predict customer churn for a telecommunications company. Predicting churn allows businesses to identify at-risk customers and take proactive retention measures.

---

## Methodology / Approach

1. **Data Loading and Exploration**
   - Loaded the Telco Customer Churn dataset.
   - Conducted exploratory data analysis (EDA) to understand data distribution, missing values, and target imbalance.
   - Visualized churn distribution.

2. **Data Preprocessing**
   - Separated numeric and categorical features.
   - Created preprocessing pipelines:
     - Numeric: Imputation (median) + Scaling (StandardScaler)
     - Categorical: Imputation (constant 'missing') + One-Hot Encoding
   - Ensured `TotalCharges` is numeric.

3. **Modeling**
   - Built two baseline models using pipelines:
     - Logistic Regression
     - Random Forest Classifier
   - Evaluated baseline performance using accuracy and ROC-AUC.

4. **Hyperparameter Tuning**
   - Performed GridSearchCV for both models to find the best parameters.
   - Optimized for ROC-AUC metric.

5. **Evaluation**
   - Evaluated the best models on the test set.
   - Metrics reported:
     - Accuracy
     - ROC-AUC
     - Classification report
     - Confusion matrix
     - ROC curves
   - Feature importance plotted for Random Forest.

6. **Model Selection**
   - Compared models using ROC-AUC and test accuracy.
   - Random Forest was selected as the final model due to superior ROC-AUC.

7. **Exporting & Deployment Readiness**
   - Saved the entire pipeline (`customer_churn_pipeline.pkl`) for production use.
   - Saved model metadata (`model_metadata.json`) for reproducibility.
   - Developed prediction function to handle new customer data.
   - Performed production readiness tests:
     - Missing values handling
     - Unknown category handling

---

## Key Results / Observations

- **Churn Rate:** Approximately 27% of customers are likely to churn.
- **Baseline Model Performance:**
  - Logistic Regression Test Accuracy: ~X%
  - Random Forest Test Accuracy: ~Y%
- **ROC-AUC Score:**
  - Logistic Regression: ~X.XXX
  - Random Forest: ~Y.XXX
- **Top Predictive Features (Random Forest):**
  - Tenure
  - Contract type
  - MonthlyCharges
  - PaymentMethod
- **Final Model:** Random Forest Pipeline
  - Includes preprocessing, feature transformation, and classifier.
  - Ready for deployment.

---

## Usage

```python
import pandas as pd
import joblib

# Load pipeline
model = joblib.load('models/customer_churn_pipeline.pkl')

# Make predictions
sample_data = pd.read_csv('data/sample_data.csv')
predictions = model.predict(sample_data)
probabilities = model.predict_proba(sample_data)
Requirements

Python 3.8+

pandas

numpy

scikit-learn

matplotlib

seaborn

joblib
