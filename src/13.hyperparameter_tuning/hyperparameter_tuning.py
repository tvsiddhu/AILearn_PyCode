# Hyperparameter tuning in Python
# Load libraries
import pandas as pd
from sklearn import linear_model

# 1. Extracting a Logistic Regression model

# Load data
credit_card_df = pd.read_csv('../../data/13.hyperparameter_tuning/credit-card-full.csv')

# Create feature matrix and target vector
X = credit_card_df.iloc[:, 1:-1]
y = credit_card_df['default payment next month']

# Split data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create logistic regression
log_reg_clf = linear_model.LogisticRegression()

# Create a list of original variable names from the training DataFrame
original_variables = X_train.columns

# Extract the coefficients of the logistic regression estimator
model_coefficients = log_reg_clf.fit(X_train, y_train).coef_[0]

# Create a DataFrame of the variables and coefficients & print it out
coefficient_df = pd.DataFrame({'Variable': original_variables, 'Coefficient': model_coefficients})
print(coefficient_df)

# Print out the top 3 positive variables
top_three_df = coefficient_df.sort_values(by='Coefficient', axis=0, ascending=False)[0:3]
print(top_three_df)
