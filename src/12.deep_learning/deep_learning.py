# coding the forward propagation algorithm
import numpy as np

#1. Forward propagation algorithm
#import the input data and weights
print("Exercise 1: Forward propagation algorithm")

input_data = np.array([3, 5])
print("Input data:", input_data)

weights = {'node_0': np.array([2, 4]), 'node_1': np.array([ 4, -5]), 'output': np.array([2, 7])}
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

#--------------------------------------------------------------

#2. Activation function (ReLU)

print("\nExercise 2: Activation function (ReLU)")

def relu(input):
    '''Define the relu function'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)

    # Return the value just calculated
    return(output)

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

#--------------------------------------------------------------
#3. Applying the network to many observations/rows of data

print("\nExercise 3: Applying the network to many observations/rows of data")

input_data = [np.array([3, 5]), np.array([ 1, -1]), np.array([0, 0]), np.array([8, 4])]
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
    return(model_output)

# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print("Predictions for each row of input data:", results)

#--------------------------------------------------------------
#4. Multi-layer neural networks

print("\nExercise 4: Multi-layer neural networks")

# Define the input data
input_data = np.array([3, 5])
print("Input data:", input_data)

# Define the weights
weights = {'node_0_0': np.array([2, 4]), 'node_0_1': np.array([ 4, -5]), 'node_1_0': np.array([-1, 2]), 'node_1_1': np.array([1, 2]), 'output': np.array([2, 7])}
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
    return(model_output)

# Make predictions using the network
output = predict_with_network(input_data)
print("Predictions for input data:", output)
