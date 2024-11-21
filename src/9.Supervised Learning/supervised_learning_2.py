# Machine learning with scikit-learn
# Supervised learning - Regression & Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold

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
from sklearn.linear_model import Ridge

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
from sklearn.linear_model import Lasso

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

