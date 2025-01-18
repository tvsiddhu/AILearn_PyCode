# Introduction to TensorFlow in Python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import constant

# 1. Defining data as constants
# -----------------------------------------------------------------
print("\n1. Defining data as constants")
print('-----------------------------------------------------------------')

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

# 13. Set up a linear regression
print("\n13. Set up a linear regression")
print('-----------------------------------------------------------------')

price_log = np.log(price)
print(price_log.shape)

size = np.array(housing['sqft_living'], np.float32)

size_log = np.log(size)
print(size_log.shape)

# Define the targets and features
price_log = Variable(price_log)
size_log = Variable(size_log)


# Define the linear regression model
def linear_regression(intercept, slope, features=size_log):
    return intercept + slope * features


# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features=size_log, targets=price_log):
    # Set the predicted values
    predictions = linear_regression(intercept, slope, features)
    # Return the mean squared error loss
    return keras.losses.mean_squared_error(targets, predictions)


# Compute the loss for different slope and intercept values
print("Loss function for slope 0.1 and intercept 0.1:", loss_function(0.1, 0.1).numpy())
print("Loss function for slope 0.1 and intercept 0.5", loss_function(0.1, 0.5).numpy())

# 14. Train a linear model
print("\n14. Train a linear model")
print('-----------------------------------------------------------------')


def plot_results(intercept, slope):
    size_range = np.linspace(6, 14, 100)
    price_pred = [intercept + slope * size for size in size_range]
    plt.scatter(size_log, price_log, color='black')
    plt.plot(size_range, price_pred, linewidth=3.0, color='red')
    plt.xlabel('log(size)')
    plt.ylabel('log(price)')
    plt.title('Scatterplot of data and regression line')
    plt.show()


# Initialize an adam optimizer
opt = keras.optimizers.Adam(0.5)


def loss_function(intercept, slope, features=size_log, targets=price_log):
    # Set the predicted values
    predictions = linear_regression(intercept, slope, features)
    # Return the mean squared error loss
    return keras.losses.mean_squared_error(targets, predictions)


intercept = Variable(10.0, float32)
slope = Variable(0.5, float32)

for j in range(100):
    with tf.GradientTape() as tape:
        loss_value = loss_function(intercept, slope)
    grads = tape.gradient(loss_value, [intercept, slope])
    opt.apply_gradients(zip(grads, [intercept, slope]))

    if j % 10 == 0:
        print(loss_function(intercept, slope).numpy())

plot_results(intercept, slope)

# 15. Multiple linear regression
print("\n15. Multiple linear regression")
print('-----------------------------------------------------------------')

bedrooms = np.array(housing['bedrooms'], np.float32)

print_results = lambda params: print('intercept: {:0.2f}, slope_1: {:0.2f}, slope_2: {:0.2f}'.format(*params.numpy()))


# Define the linear regression model
def linear_regression(params, feature1=size_log, feature2=bedrooms):
    return params[0] + feature1 * params[1] + feature2 * params[2]


def loss_function(params, targets=price_log, feature1=size_log, feature2=bedrooms):
    # Set the predicted values
    predictions = linear_regression(params, feature1, feature2)
    # Return the mean absolute error loss
    return keras.losses.mean_absolute_error(targets, predictions)


# Define the optimize operation
opt = keras.optimizers.Adam()

# Initialize the parameters
params = Variable([0.1, 0.05, 0.02], float32)

# Perform the optimization
for j in range(10):
    with tf.GradientTape() as tape:
        # Compute the loss
        loss = loss_function(params)
    # Compute the gradients
    grads = tape.gradient(loss, [params])
    # Perform minimization
    opt.apply_gradients(zip(grads, [params]))
    print_results(params)

# 16. Preparing to batch train
print("\n16. Preparing to batch train")
print('-----------------------------------------------------------------')

# Define the intercept and slope
intercept = Variable(10.0, float32)
slope = Variable(0.5, float32)


# Define the model
def linear_regression(intercept, slope, features):
    # Define the predicted values
    return intercept + slope * features


# Define the loss function
def loss_function(intercept, slope, targets, features):
    # Define the predicted values
    predictions = linear_regression(intercept, slope, features)
    # Return the mean squared error loss
    return keras.losses.mean_squared_error(targets, predictions)


# 17. Training a linear model in batches
print("\n17. Training a linear model in batches")
print('-----------------------------------------------------------------')

# Initialize adam optimizer
opt = keras.optimizers.Adam()

# Load data in batches
for batch in pd.read_csv(data_path, chunksize=100):
    size_batch = np.array(batch['sqft_living'], np.float32)
    price_batch = np.array(batch['price'], np.float32)
    size_log = np.log(size_batch)
    price_log = np.log(price_batch)

with tf.GradientTape() as tape:
    loss_value = loss_function(intercept, slope, price_log, size_log)
grads = tape.gradient(loss_value, [intercept, slope])
opt.apply_gradients(zip(grads, [intercept, slope]))

# Print the intercept and slope
print("intercept: {:0.2f}, slope: {:0.2f}".format(intercept.numpy(), slope.numpy()))
