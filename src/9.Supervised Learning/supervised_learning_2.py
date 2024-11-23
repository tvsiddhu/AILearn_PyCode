# Machine learning with scikit-learn
# Supervised learning - Regression & Linear Regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import class_weight

# Load the sales dataset
sales_df = pd.read_csv('../../data/supervised_learning/advertising_and_sales.csv')

# Create X from the radio column's values
X = sales_df["radio"].values
# Create y from the sales column's values
y = sales_df["sales"].values

# Reshape X and y
X = X.reshape(-1, 1)
# y = y.reshape(-1, 1)

# Check the shape of X and y
print(X.shape)
print(y.shape)

# Create a linear regression model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Check the model's predictions
predictions = reg.predict(X)

print("Predictions: ", predictions[:5])

# Plot the data
plt.scatter(X, y, color='blue')
plt.plot(X, predictions, color='red', linewidth=3)
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Show the plot
plt.show()

# Fit and predict for regression

# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df.sales.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate a Linear regression model
reg_all = LinearRegression()

# Fit the model to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data
y_pred = reg_all.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Regression Performance
# Compute and print the R^2 and RMSE
r_squared = reg_all.score(X_test, y_test)

# Compute the mean squared error of the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the R^2 and RMSE
print("R^2: {}".format(r_squared))
print("Root Mean Squared Error: {}".format(rmse))

# Cross-validation
# Create a K-Fold cross-validation object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print the cross-validation scores
print("Cross-validation scores: {}".format(cv_scores))

# Analyzing the cross-validation results

# Print the mean
print("Average 6-Fold CV Score: {}".format(np.mean(cv_scores)))

# Print the standard deviation
print("Standard deviation of 6-Fold CV scores: {}".format(np.std(cv_scores)))

# Print the 95% confidence interval
print("95% confidence interval: {}".format(np.percentile(cv_scores, [2.5, 97.5])))

# Regularized regression

# Import Ridge


alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []

for alpha in alphas:
    # Instantiate a Ridge regressor
    ridge = Ridge(alpha=alpha)

    # Fit the model to the data
    ridge.fit(X_train, y_train)

    # Obtain R^2 score
    score = ridge.score(X_test, y_test)
    ridge_scores.append(score)
print("Ridge Scores: ", ridge_scores)

# Lasso regression for feature selection


# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print("Lasso Coefficients: ", lasso_coef)
plt.bar(range(len(sales_df.columns) - 1), lasso_coef)
plt.xticks(range(len(sales_df.columns) - 1), sales_df.columns[:-1], rotation=45)
plt.ylabel("Coefficients")
plt.show()

# Assessing a diabetes prediction classifier

# Load the diabetes dataset
diabetes_df = pd.read_csv('../../data/supervised_learning/diabetes_clean.csv')

knn = KNeighborsClassifier(n_neighbors=6)

# Create arrays for the features and the response variable
y = diabetes_df['diabetes'].values
X = diabetes_df.drop('diabetes', axis=1).values

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("Classification Report: ")
print(classification_report(y_test, y_pred))

# Building a logistic regression model


# Instantiate a Logistic Regression model

logreg = LogisticRegression()

# Fit the model to the data
logreg.fit(X_train, y_train)

# Predict the probabilities of each individual in the test set having a diabetes diagnosis, storing the array of
# positive probabilities as y_pred_probs.

y_pred_probs = logreg.predict_proba(X_test)[:, 1]

print("Predicted Probabilities: ", y_pred_probs[:10])

# The ROC curve

# Generate ROC Curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()

# AUC computation
# Import the roc_auc_score function

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_probs)))

# Calculate the confusion matrix and classification report
y_pred = logreg.predict(X_test)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("Classification Report: ")
print(classification_report(y_test, y_pred))

# Hyperparameter tuning with GridSearchCV

# Set up the hyperparameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)

# Fit to the training set
lasso_cv.fit(X_train, y_train)

# Predict the labels of the test set
print("Tuned lasso parameters: {}".format(lasso_cv.best_params_))
print(" Tuned lasso score: {}".format(lasso_cv.best_score_))

# Hyperparameter tuning with RandomizedSearchCV

# Import RandomizedSearchCV

# Create the parameter grid
# Check the parameter space
params = {
    "penalty": ["l1", "l2"],
    "tol": np.linspace(0.0001, 1.0, 50),  # Make sure to include 'tol' in the params
    "C": np.linspace(0.1, 1.0, 50),
    "class_weight": ["balanced", {0:0.8, 1:0.2}]
}

# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)

# Fit it to the data
logreg_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))
