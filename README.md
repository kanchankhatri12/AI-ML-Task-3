# AI/ML Internship Tasks

## Task 3: Heart Disease Prediction


## Task Objective

Build a machine learning model to predict whether a person is at risk of heart disease based on their health data. The goal is to identify key risk factors and create an accurate classification system for early detection of heart disease.


##  Dataset Used

- **Dataset Name:** Heart Disease UCI Dataset
- **Source:** Kaggle / UCI Machine Learning Repository
- **Description:** The dataset contains 14 attributes related to cardiovascular health.

| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| sex | Gender (0 = female, 1 = male) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (0/1) |
| restecg | Resting ECG results (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (0/1) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) |
| target | Diagnosis (0 = no disease, 1 = disease) |


##  Models Applied

The following classification models were trained and evaluated:

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear model for binary classification |
| Decision Tree Classifier | Tree-based model for interpretable predictions |

### Model Pipeline:

1. Data cleaning and missing value treatment
2. Exploratory Data Analysis (EDA)
3. Train-test split (80/20)
4. Model training and evaluation


##  Key Results & Findings

### Model Performance:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 1.0 |
| Decision Tree | 1.0 |

### Top 5 Most Important Features (Decision Tree):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | cp (Chest Pain Type) | 0.262 |
| 2 | ca (Major Vessels) | 0.147 |
| 3 | age | 0.096 |
| 4 | thal (Thalassemia) | 0.089 |
| 5 | oldpeak | 0.089 |

### Key Insights:

- **Chest Pain Type (cp)** is the strongest predictor of heart disease
- **Number of major vessels (ca)** and **thalassemia (thal)** are also highly significant
- The model achieved perfect accuracy (1.0) on the test set, indicating strong predictive capability
- ROC curves confirm excellent model discrimination between healthy and diseased patients


##  Visualizations Included

- Target distribution (countplot)
- Correlation heatmap of all features
- Age distribution by disease status (boxplot)
- Confusion matrix
- ROC curve with AUC score


## Tools & Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Conclusion

The Heart Disease Prediction model successfully identifies patients at risk of heart disease with high accuracy. Chest pain type emerged as the most significant predictor, followed by the number of major vessels and thalassemia. These findings align with medical knowledge and demonstrate the potential of machine learning in healthcare diagnostics.

##  How to Run
1. Clone the repository
2. Install required dependencies:
3. Run the Jupyter notebook: `HeartDisease.ipynb`

## Future Improvements

- Implement additional models (Random Forest, XGBoost)
- Perform hyperparameter tuning
- Cross-validation for more robust evaluation
- Deploy model as a web API for real-time predictions.
