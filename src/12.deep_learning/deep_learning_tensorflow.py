# Introduction to TensorFlow in Python

# 1. Defining data as constants
# -----------------------------------------------------------------
print("\n1. Defining data as constants")
print('-----------------------------------------------------------------')

import pandas as pd
from tensorflow import constant

# Import the data
df = pd.read_csv('../../data/12.deep_learning/credit_numpy_array.csv', header=0, index_col=0)
print(df.head())

# Convert the dataframe to a Numpy array
credit_numpy = df.values
credit_constant = constant(credit_numpy)

# Print constant datatype
print('\n The datatype is:', credit_constant.dtype)

# Print constant shape
print('\n The shape is:', credit_constant.shape)

# 2. Defining variables
from tensorflow import Variable

# -----------------------------------------------------------------
print("\n2. Defining variables")
print('-----------------------------------------------------------------')
# Define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])

# Print the variable A1
print('\n A1:', A1)
print('\n The datatype for A1 is:', A1.dtype)

# Convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()

# Print B1
print('\n B1:', B1)
print('\n The datatype for B1 is:', B1.dtype)

# 3. Performing element-wise multiplication
print("\n3. Performing element-wise multiplication")
print('-----------------------------------------------------------------')
from tensorflow import multiply, ones_like, matmul

# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the same shape as A1 and A23 respectively
B1 = ones_like(A1)
B23 = ones_like(A23)

# Perform element-wise multiplication
C1 = multiply(A1, B1)
C23 = multiply(A23, B23)

# Print the tensors C1 and C23
print('\n C1: {}'.format(C1.numpy()))
print('\n C23: {}'.format(C23.numpy()))

# 4. Making predictions with matrix multiplication
print("\n4. Making predictions with matrix multiplication")
print('-----------------------------------------------------------------')

# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
billpred = matmul(features, params)

# Compute and print the error
error = bill - billpred
print('\n The error is:', error.numpy())

# 5. Summing over tensor dimensions
print("\n5. Summing over tensor dimensions")
print('-----------------------------------------------------------------')

from tensorflow import reduce_sum, reduce_max, reduce_min

wealth = constant([[11, 50], [7, 2], [4, 60], [3, 0], [25, 10]])
# Sum over dimension 0 and 1 of wealth
total_per_col = reduce_sum(wealth, 0).numpy()
total_per_row = reduce_sum(wealth, 1).numpy()

print("Total per col:", total_per_col)
print("Total per row:", total_per_row)

# Largest/smallest of columns (sum along rows, dimension 0)
largest_columns = reduce_max(wealth, axis=0).numpy()
smallest_columns = reduce_min(wealth, axis=0).numpy()

# Largest/smallest of rows (sum along columns, dimension 1)
largest_rows = reduce_max(wealth, axis=1).numpy()
smallest_rows = reduce_min(wealth, axis=1).numpy()

print("Largest of columns:", largest_columns)
print("Smallest of columns:", smallest_columns)
print("Largest of rows:", largest_rows)
print("Smallest of rows:", smallest_rows)

# 6. Reshaping tensors
print("\n6. Reshaping tensors")
print('-----------------------------------------------------------------')

from tensorflow import reshape
import pandas as pd

gray_image = pd.read_csv('../../data/12.deep_learning/gray_tensor.csv', header=0, index_col=0)
gray_tensor = gray_image.to_numpy()

print("gray tensor shape:", gray_tensor.shape)

# reshape the gray_tensor from a 28x28 matrix into a 784x1 vector
gray_vector = reshape(gray_tensor, (784, 1))
print(gray_vector.shape)

colour_image = pd.read_csv('../../data/12.deep_learning/colour_tensor.csv', header=None)
colour_tensor = colour_image.to_numpy()
print("colour tensor shape:", colour_tensor.shape)
colour_tensor_reshaped = colour_tensor.flatten().reshape(-1, 1)
print("colour tensor reshaped shape:", colour_tensor_reshaped.shape)

colour_vector = reshape(colour_tensor, (2352, 1))
print("colour vector shape:", colour_vector.shape)

# 7. Optimizing with gradients
print("\n7. Optimizing with gradients")
print('-----------------------------------------------------------------')

from tensorflow import GradientTape, multiply, Variable
import numpy as np


def compute_gradient(x0):
    # Define x as a variable equal to x0
    x = Variable(x0)
    with GradientTape() as tape:
        tape.watch(x)
        # Define y using the multiply operation
        y = multiply(x, x)
    # Return the gradient of y with respect to x
    return tape.gradient(y, x).numpy()


# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))

# 8. Working with image data
print("\n8. Working with image data")
print('-----------------------------------------------------------------')

letter = np.array([[1.0, 0, 1.0], [1., 1., 0], [1., 0, 1.]])
model = np.array([[1., 0., -1.]])

model = reshape(model, (3, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print("Prediction: {}".format(prediction.numpy()))

# INPUT DATA
# 9. Load data using pandas
print("\n9. Load data using pandas")
print('-----------------------------------------------------------------')

from tensorflow import cast

# Load the data
data_path = '../../data/12.deep_learning/kc_house_data.csv'

# Load data using read_csv
housing = pd.read_csv(data_path)

# Print the price column of housing
print(housing['price'])

# 10. Setting the data type
print("\n10. Setting the data type")
print('-----------------------------------------------------------------')

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# Define waterfront as a Boolean using cast
waterfront = cast(housing['waterfront'], bool)

# Print price and waterfront
print("Price:", price)
print("Waterfront:", waterfront)

# 11. Loss functions in TensorFlow
print("\n11. Loss functions in TensorFlow")
print('-----------------------------------------------------------------')

import keras

print("price shape:", price.shape)

df = pd.read_csv('../../data/12.deep_learning/predictions.csv', header=0).to_numpy()
predictions = df[:, 0]
print("predictions shape:", predictions.shape)

# Compute the mean squared error (mse)
loss = keras.losses.mean_squared_error(price, predictions)

# Print the mean squared error (mse)
print("Loss function using mean squared error:", loss.numpy())

# print the mean absolute error (mae)
loss = keras.losses.mean_absolute_error(price, predictions)
print("Loss function using mean absolute error:", loss.numpy())

# print the huber loss
loss = keras.losses.huber(price, predictions)
print("Loss function using huber loss:", loss.numpy())

# 12. Modifying the loss function
print("\n12. Modifying the loss function")
print('-----------------------------------------------------------------')

features = np.array([[1, 2, 3, 4, 5]], np.float32)
float32 = np.float32
targets = np.array([2, 4, 6, 8, 10], np.float32)


# Initialize a variable named scalar
scalar = Variable(1.0, float32)

# Define the model
def model(scalar, features=features):
    return scalar * features

# Define a loss function
def loss_function(scalar, features=features, targets=targets):
    # Compute the predicted values
    predictions = model(scalar, features)
    # Return the mean absolute error loss
    return keras.losses.mean_absolute_error(targets, predictions)

# Evaluate the loss function and print the loss
print(loss_function(scalar).numpy())
