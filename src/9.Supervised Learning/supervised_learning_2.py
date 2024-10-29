# Machine learning with scikit-learn
# Supervised learning - Regression & Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression



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

