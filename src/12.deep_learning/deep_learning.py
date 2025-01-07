# coding the forward propagation algorithm
import numpy as np
import pandas as pd

# 1. Forward propagation algorithm
# import the input data and weights
print("Exercise 1: Forward propagation algorithm")

input_data = np.array([3, 5])
print("Input data:", input_data)

weights = {'node_0': np.array([2, 4]), 'node_1': np.array([4, -5]), 'output': np.array([2, 7])}
print("weights:", weights)

# Calculate node 0 value: node_0_value
node_0_value = (input_data * weights['node_0']).sum()
print("Node 0 value:", node_0_value)

# Calculate node 1 value: node_1_value
node_1_value = (input_data * weights['node_1']).sum()
print("Node 1 value:", node_1_value)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_value, node_1_value])
print("Hidden layer outputs:", hidden_layer_outputs)

# Calculate output: output
output = (hidden_layer_outputs * weights['output']).sum()

# Print output
print("Forward propagation algorithm (hidden layer outputs * weights of outputs):", output)

# --------------------------------------------------------------

# 2. Activation function (ReLU)

print("\nExercise 2: Activation function (ReLU)")


def relu(input):
    '''Define the relu function'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)

    # Return the value just calculated
    return (output)


# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)
print("Node 0 output (ReLU):", node_0_output)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)
print("Node 1 output (ReLU):", node_1_output)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])
print("Hidden layer outputs (ReLU):", hidden_layer_outputs)

# Calculate model output (do not apply the ReLU function)
output = (hidden_layer_outputs * weights['output']).sum()
print("Forward propagation algorithm (hidden layer outputs * weights of outputs):", output)

# --------------------------------------------------------------
# 3. Applying the network to many observations/rows of data

print("\nExercise 3: Applying the network to many observations/rows of data")

input_data = [np.array([3, 5]), np.array([1, -1]), np.array([0, 0]), np.array([8, 4])]
print("Input data:", input_data)


# Define the predict_with_network() function
def predict_with_network(input_data_row, weights):
    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])

    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)

    # Return model output
    return (model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print("Predictions for each row of input data:", results)

# --------------------------------------------------------------
# 4. Multi-layer neural networks

print("\nExercise 4: Multi-layer neural networks")

# Define the input data
input_data = np.array([3, 5])
print("Input data:", input_data)

# Define the weights
weights = {'node_0_0': np.array([2, 4]), 'node_0_1': np.array([4, -5]), 'node_1_0': np.array([-1, 2]),
           'node_1_1': np.array([1, 2]), 'output': np.array([2, 7])}
print("Weights:", weights)


# Define the predict_with_network() function
def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output
    input_to_final_layer = (hidden_1_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)

    # Return model output
    return (model_output)


# Make predictions using the network
output = predict_with_network(input_data)
print("Predictions for input data:", output)

# --------------------------------------------------------------
# 5. Coding how weight changes affect accuracy

print("\nExercise 5: Coding how weight changes affect accuracy")

# The data point you will make a prediction for
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': np.array([2, 1]), 'node_1': np.array([1, 2]), 'output': np.array([1, 1])}

# The actual target value, used to calculate the error
target_actual = 3


# Define the predict_with_network() function
def predict_with_network(input_data, weights):
    # Calculate node 0 value
    node_0_input = (input_data * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])

    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)

    # Return model output
    return (model_output)


# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)
print("Model output with original weights:", model_output_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual
print("Error with original weights:", error_0)

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': np.array([2, 1]), 'node_1': np.array([1, 2]), 'output': np.array([1, 0])}

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)
print("Model output with new weights:", model_output_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual
print("Error with new weights:", error_1)

# Print the original error
print("Original error:", error_0)
# Print the new error
print("New error:", error_1)

# --------------------------------------------------------------
# 6. Scaling up to multiple data points

print("\nExercise 6: Scaling up to multiple data points")

# import mean_squared_error from sklearn.metrics
from sklearn.metrics import mean_squared_error

# Define the input data
input_data = [np.array([0, 3]), np.array([1, 2]), np.array([-1, -2]), np.array([4, 0])]
print("Input data:", input_data)

# Define the actual target values
target_actuals = [1, 3, 5, 7]
print("Actual target values:", target_actuals)

# Sample weights
weights_0 = {'node_0': np.array([2, 1]), 'node_1': np.array([1, 2]), 'output': np.array([1, 1])}
weights_1 = {'node_0': np.array([2, 1]), 'node_1': np.array([1., 1.5]), 'output': np.array([1., 1.5])}

# Create model_output_0
model_output_0 = []
# Create model_output_1
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals, model_output_0)
print("Mean squared error with original weights:", mse_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)
print("Mean squared error with new weights:", mse_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" % mse_0)
print("Mean squared error with weights_1: %f" % mse_1)

# --------------------------------------------------------------
# 7. Calculating slopes

print("\nExercise 7: Calculating slopes")

# Define the input data
input_data = np.array([1, 2, 3])
print("Input data:", input_data)

# Define the weights
weights = np.array([0, 2, 1])
print("Weights:", weights)

# Define the actual target value
target = 0
print("Actual target value:", target)

# Calculate the predictions: preds
preds = (input_data * weights).sum()
print("Predictions:", preds)

# Calculate the error: error
error = preds - target
print("Error:", error)

# Calculate the slope: slope
slope = 2 * input_data * error

# Print the slope
print("Slope:", slope)

# --------------------------------------------------------------
# 8. Improving model weights

print("\nExercise 8: Improving model weights")

# Set the learning rate: learning_rate
learning_rate = 0.01

# Update the weights: weights_updated
weights_updated = weights - learning_rate * slope

# Get updated predictions: preds_updated
preds_updated = (input_data * weights_updated).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print("Original error:", error)

# Print the updated error
print("Updated error:", error_updated)

# --------------------------------------------------------------
# 9. Making multiple updates to weights

print("\nExercise 9: Making multiple updates to weights")

# Import matplotlib
import matplotlib.pyplot as plt

# Define the input data
input_data = np.array([1, 2, 3])
print("Input data:", input_data)

# Define the actual target value
target = 0
print("Actual target value:", target)

# Define the weights
weights = np.array([0, 2, 1])
print("Weights:", weights)

n_updates = 20
mse_hist = []


# Define the get_slope() function
def get_slope(input_data, target, weights):
    # Calculate the predictions: preds
    preds = (input_data * weights).sum()

    # Calculate the error: error
    error = preds - target

    # Calculate the slope: slope
    slope = 2 * input_data * error

    return (slope)


# define the get_mse() function
def get_mse(input_data, target, weights):
    # Calculate the predictions: preds
    preds = (input_data * weights).sum()

    # Calculate the mean squared error: mse
    mse = mean_squared_error([target], [preds])

    return (mse)


# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)

    # Update the weights: weights
    weights = weights - 0.01 * slope

    # Calculate the mean squared error: mse
    mse = get_mse(input_data, target, weights)
    mse_hist.append(mse)

# Plot the mse_hist
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

# --------------------------------------------------------------

# 10. Specifying the model

print("\nExercise 10: Specifying the model")

# Import the hourly wage data


df = pd.read_csv('../../data/12.deep_learning/hourly_wages.csv', encoding='latin1')

predictors = df.drop('wage_per_hour', axis=1).values
target = df['wage_per_hour'].values

# Import the Sequential model and Dense layer

from keras import models, layers, utils

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = models.Sequential([
    layers.Input(shape=(n_cols,)),
    layers.Dense(50, activation='relu')
])

# # Add the first layer
# model.add(layers.Dense(50, activation='relu', input_shape=(n_cols,)))
#
# # Add the second layer
# model.add(layers.Dense(32, activation='relu'))

# Add the output layer
model.add(layers.Dense(1))

# --------------------------------------------------------------
# 11. Compiling the model

print("\nExercise 11: Compiling the model")

# Specify the model
n_cols = predictors.shape[1]
model = models.Sequential()
model.add(layers.Dense(50, activation='relu', input_shape=(n_cols,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model is compiled
print("\nModel summary:" )
print(model.summary())
print("\nLoss function:" + model.loss)

# --------------------------------------------------------------
# 12. Fitting the model

print("\nExercise 12: Fitting the model")

# Fit the model
model.fit(predictors, target)

# --------------------------------------------------------------
# 13. Last steps in classification models

print("\nExercise 13: Last steps in classification models")

# Import the data
# df = pd.read_csv('../../data/12.deep_learning/titanic_all_numeric.csv', encoding='latin1')
df = pd.read_csv('../../data/12.deep_learning/titanic_all_numeric.csv', encoding='latin1', dtype=str)

# predictors = df.drop('survived', axis=1).values

predictors = df.drop('survived', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0).values

# Convert the target to categorical: target
target = utils.to_categorical(df['survived'])

# Specify the model
n_cols = predictors.shape[1]

# Set up the model
model = models.Sequential()

# Add the first layer
model.add(layers.Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# Fit the model
model.fit(predictors, target)

# --------------------------------------------------------------
# 14. Making predictions

print("\nExercise 14: Making predictions")

# Not including this code since the underlying data is not available as part of the course

# # Import the new data
# pred_data = pd.read_csv('../../data/12.deep_learning/pred_data.csv', encoding='latin1', dtype=str)
# pred_data = pred_data.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
#
# model = models.Sequential([layers.Input(shape=(32,11)), layers.Dense(50, activation='relu')])
#
# # Add the output layer
# model.add(layers.Dense(2, activation='softmax'))
#
# # Compile the model
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Fit the model
# model.fit(predictors, target)
#
# # Calculate predictions: predictions
# predictions = model.predict(pred_data)
#
# # Calculate predicted probability of survival: predicted_prob_true
# predicted_prob_true = predictions[:, 1]
#
# # print predicted_prob_true
# print("Predicted probability of survival:", predicted_prob_true)

# --------------------------------------------------------------
# 15. Changing optimization parameters

