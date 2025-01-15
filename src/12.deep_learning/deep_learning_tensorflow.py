# Introduction to TensorFlow in Python

# 1. Defining data as constants
# -----------------------------------------------------------------
print("\n1. Defining data as constants")
print('-----------------------------------------------------------------')

from tensorflow import constant
import pandas as pd

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

