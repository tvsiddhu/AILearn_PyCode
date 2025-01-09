# Introduction to TensorFlow in Python

#1. Defining data as constants
from tensorflow import constant
import pandas as pd

# Import the data
df = pd.read_csv('../../data/12.deep_learning/credit_numpy_array.csv', header=0,index_col=0)
print(df.head())

# Convert the dataframe to a Numpy array
credit_numpy = df.values
credit_constant = constant(credit_numpy)

# Print constant datatype
print('\n The datatype is:', credit_constant.dtype)

# Print constant shape
print('\n The shape is:', credit_constant.shape)
