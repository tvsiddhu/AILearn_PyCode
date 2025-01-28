# Introduction to TensorFlow in Python
import ssl
import warnings

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.utils import to_categorical
from tensorflow import GradientTape, multiply, Variable
from tensorflow import cast
from tensorflow import constant
from tensorflow import ones_like, matmul
from tensorflow import reduce_sum, reduce_max, reduce_min
from tensorflow import reshape

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
# Register date converters to avoid warning
pd.plotting.register_matplotlib_converters()

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

# 18. The linear algebra of dense layers
print("\n18. The linear algebra of dense layers")
print('-----------------------------------------------------------------')

# Define the features and targets
borrower_features = np.array([[2., 2., 43]], np.float32)

# Initialize bias1
bias1 = Variable(1.0)

# Initialize weights1 as 3x2 variable of ones
weights1 = Variable(tf.ones((3, 2)))

# Perform matrix multiplication of borrower_features and weights1
product1 = matmul(borrower_features, weights1)

# Apply sigmoid activation function to product1 + bias1
dense1 = keras.activations.sigmoid(product1 + bias1)

# Print shape of dense1
print("\n dense1's output shape: ", dense1.shape)

# Initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(tf.ones((2, 1)))

# Perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1, weights2)

# Apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2 + bias2)
print('\n prediction: {}'.format(prediction.numpy()[0, 0]))
print('\n actual: 1')

# 19. The low-level approach with multiple examples
print("\n19. The low-level approach with multiple examples")
print('-----------------------------------------------------------------')

borrower_features = np.array([[3., 3., 23], [2., 1., 24], [1., 1., 49], [1., 1., 49], [2., 1., 29]], np.float32)
bias1 = Variable(1.0)
weights1 = Variable(tf.ones((3, 2)))

# Compute the product of borrower_features and weights1
products1 = matmul(borrower_features, weights1)
print("\n shape of products1:", products1.shape)

# Apply a sigmoid activation function to products1 + bias1
dense1 = keras.activations.sigmoid(products1 + bias1)

# Print the shapes of borrower_features, weights1, bias1, and dense1
print("\n shape of borrower_features:", borrower_features.shape)
print("\n shape of weights1:", weights1.shape)
print("\n shape of bias1:", bias1.shape)
print("\n shape of dense1:", dense1.shape)

# 20. Using the dense layer operation
print("\n20. Using the dense layer operation")
print('-----------------------------------------------------------------')

# Define the features and the targets
borrower_features = pd.read_csv('../../data/12.deep_learning/borrower_features_100.csv', header=None).to_numpy()
borrower_features = np.array(borrower_features, np.float32)

# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print the shapes of dense1, dense2, and predictions
print("\n shape of dense1:", dense1.shape)
print("\n shape of dense2:", dense2.shape)
print("\n shape of predictions:", predictions.shape)

# 21. Binary classification problems
print("\n21. Binary classification problems")
print('-----------------------------------------------------------------')

credit_data = pd.read_csv('../../data/12.deep_learning/uci_credit_card.csv', header=0).to_numpy()
bill_amount_1 = credit_data[:, 12]
bill_amount_2 = credit_data[:, 13]
bill_amount_3 = credit_data[:, 14]

bill_amounts = np.column_stack((bill_amount_1, bill_amount_2, bill_amount_3))
bill_amounts = np.array(bill_amounts, np.float32)
default = credit_data[:, 24]

# Construct input layer from features
inputs = constant(bill_amounts)

# Define first dense layer
dense1 = keras.layers.Dense(3, activation='relu')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(2, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print error for first five examples
error = default[:5] - outputs.numpy()[:5]
print("\n error for first five examples:", error)

# 22. Multiclass classification problems
print("\n22. Multiclass classification problems")
print('-----------------------------------------------------------------')

# Select the features and targets
borrower_features = credit_data[:, 12:22]
borrower_features = np.array(borrower_features, np.float32)

targets = credit_data[:, 24]

# Construct input layer from borrower features
inputs = constant(borrower_features)

# Define first dense layer
dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(8, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(6, activation='softmax')(dense2)

# Print first five predictions
print(outputs.numpy()[:5])

# 23. The dangers of local minima
print("\n23. The dangers of local minima")
print('-----------------------------------------------------------------')

# Initialize x_1 and x_2
x_1 = Variable(6.0, float32)
x_2 = Variable(0.3, float32)


# Example loss function
def loss_function(x):
    return tf.square(x - 3)  # Example: minimize (x - 3)^2


opt = keras.optimizers.SGD(learning_rate=0.1)

for j in range(100):
    # Compute gradients for both x_1 and x_2
    with tf.GradientTape() as tape:
        loss_x1 = loss_function(x_1)
        loss_x2 = loss_function(x_2)
        total_loss = loss_x1 + loss_x2  # Combine the losses (optional)

    grads = tape.gradient(total_loss, [x_1, x_2])
    opt.apply_gradients(zip(grads, [x_1, x_2]))

print(x_1.numpy(), x_2.numpy())

# 24. Avoiding local minima
print("\n24. Avoiding local minima")
print('-----------------------------------------------------------------')

# Initialize variables
x_1 = Variable(0.05, dtype=tf.float32)
x_2 = Variable(0.05, dtype=tf.float32)

# Define optimizers with different momentum values
opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.00)

# Optimization loop
for j in range(100):
    # Compute gradients and apply updates for x_1
    with tf.GradientTape() as tape1:
        loss_x1 = loss_function(x_1)
    grads_x1 = tape1.gradient(loss_x1, [x_1])
    opt_1.apply_gradients(zip(grads_x1, [x_1]))

    # Compute gradients and apply updates for x_2
    with tf.GradientTape() as tape2:
        loss_x2 = loss_function(x_2)
    grads_x2 = tape2.gradient(loss_x2, [x_2])
    opt_2.apply_gradients(zip(grads_x2, [x_2]))

# Print the final values of x_1 and x_2
print(x_1.numpy(), x_2.numpy())

# 25. Initialization in TensorFlow
print("\n25. Initialization in TensorFlow")
print('-----------------------------------------------------------------')

# Define the layer 1 weights
w1 = Variable(tf.random.normal([23, 7]))

# Initialize the layer 1 bias
b1 = Variable(tf.ones([7]))

# Define the layer 2 weights
w2 = Variable(tf.random.normal([7, 1]))

# Define the layer 2 bias
b2 = Variable(0.0)

# 26. Defining the model and loss function
print("\n26. Defining the model and loss function")
print('-----------------------------------------------------------------')


# Define the model
def model(w1, b1, w2, b2, features=borrower_features):
    # Apply relu activation functions to layer 1
    layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout
    dropout = keras.layers.Dropout(0.25)(layer1)
    return keras.activations.sigmoid(matmul(dropout, w2) + b2)


# Define the loss function
def loss_function(w1, b1, w2, b2, features=borrower_features, targets=targets):
    predictions = model(w1, b1, w2, b2)
    # Pass targets and predictions to the cross entropy loss
    return keras.losses.binary_crossentropy(targets, predictions)


# 27. Training neural networks with TensorFlow
print("\n27. Training neural networks with TensorFlow")
print('-----------------------------------------------------------------')

test_features_data = pd.read_csv('../../data/12.deep_learning/nn_test_features.csv', header=0).to_numpy()
test_features = test_features_data[:, 1:]
test_features = np.array(test_features, np.float32)
test_features = constant(test_features)

test_targets_data = pd.read_csv('../../data/12.deep_learning/nn_test_targets.csv', header=0).to_numpy()
test_target = test_targets_data[:, 1:]
test_targets = np.array(test_target, np.float32)

opt = keras.optimizers.SGD(learning_rate=0.1)


# Define the model
def model(w1, b1, w2, b2, features=test_features):
    # Apply relu activation functions to layer 1
    layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout
    dropout = keras.layers.Dropout(0.25)(layer1)
    return keras.activations.sigmoid(matmul(dropout, w2) + b2)


# Define the loss function
def loss_function(w1, b1, w2, b2, features=test_features, targets=test_targets):
    predictions = model(w1, b1, w2, b2)
    # Pass targets and predictions to the cross entropy loss
    return keras.losses.binary_crossentropy(targets, predictions)


# Train the model
for j in range(100):
    with tf.GradientTape() as tape:
        loss_value = loss_function(w1, b1, w2, b2, test_features, test_targets)
    grads = tape.gradient(loss_value, [w1, b1, w2, b2])
    opt.apply_gradients(zip(grads, [w1, b1, w2, b2]))

# Make predictions with model using test_features
model_predictions = model(w1, b1, w2, b2, test_features)

# Flatten the model_predictions to match the shape of test_targets
model_predictions = tf.reshape(model_predictions, [-1])

# Construct the confusion matrix
confusion_matrix = tf.math.confusion_matrix(test_targets, model_predictions)
print("\n Confusion matrix: ", confusion_matrix.numpy())

# 28. The sequential model in Keras
print("\n28. The sequential model in Keras")
print('-----------------------------------------------------------------')

# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Print the model architecture
print("Model architecture: ", model.summary())

# 29. Compiling a sequential model
print("\n29. Compiling a sequential model")
print('-----------------------------------------------------------------')

# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(784,)))

# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print("Model summary: ", model.summary())

# Defining a multiple input model
print("\n30. Defining a multiple input model")
print('-----------------------------------------------------------------')

# Define the first input
m1_inputs = keras.Input(shape=(784,))
# Define the second input
m2_inputs = keras.Input(shape=(784,))

# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print("Model summary: ", model.summary())

# 31. Training with Keras
print("\n31. Training with Keras")
print('-----------------------------------------------------------------')

ssl._create_default_https_context = ssl._create_unverified_context

# Define sign language digits data
# Load EMNIST dataset

emnist = keras.datasets.mnist
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = emnist.load_data()

# Merge training and test sets for a larger dataset
x_data = np.concatenate((x_train, x_test), axis=0)
y_data = np.concatenate((y_train, y_test), axis=0)

# Mapping MNIST digits to approximate letters
letter_mapping = {4: 0, 8: 1, 6: 2, 0: 3}  # 4 → A, 8 → B, 6 → C, 0 → D

# Filter only images corresponding to these letters
mask = np.isin(y_data, list(letter_mapping.keys()))
x_filtered = x_data[mask]
y_filtered = y_data[mask]

# Convert labels to range (0-3) based on letter_mapping
y_filtered = np.array([letter_mapping[y] for y in y_filtered])

# Select only 2000 samples
num_samples = 2000
x_filtered = x_filtered[:num_samples]
y_filtered = y_filtered[:num_samples]

# Normalize images (scale pixel values between 0 and 1)
x_filtered = x_filtered.astype('float32') / 255.0

# Reshape images for CNN input (28x28x1)
x_filtered = x_filtered.reshape(-1, 28, 28, 1)

# Convert labels to categorical format for classification
y_filtered = to_categorical(y_filtered, num_classes=4)

# Assign to final variables
sign_language_features = x_filtered
sign_language_labels = y_filtered

# Display sample images to verify correctness
fig, axes = plt.subplots(1, 4, figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.imshow(sign_language_features[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {np.argmax(sign_language_labels[i])} (A/B/C/D)")
    ax.axis('off')
plt.show()

# Print shape confirmation
print(f"Features shape: {sign_language_features.shape}")  # Expected: (2000, 28, 28, 1)
print(f"Labels shape: {sign_language_labels.shape}")  # Expected: (2000, 4)

sign_language_features = sign_language_features.reshape(-1, 784)
print(f"Reshaped Features shape: {sign_language_features.shape}")  # Expected: (, 784)

# Define a Keras sequential model
model = keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)

# 32. Metrics and validation with Keras
print("\n32. Metrics and validation with Keras")
print('-----------------------------------------------------------------')

# Define a Keras sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.1)

# 33. Overfitting detection
print("\n33. Overfitting detection")
print('-----------------------------------------------------------------')

# Define a Keras sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024, activation='relu', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
              metrics=['accuracy'])

# Add the number of epochs and the validation split
history = model.fit(sign_language_features, sign_language_labels, epochs=50, validation_split=0.5)

# 34. Preparing to train with Estimators
print("\n34. Preparing to train with Estimators")
print('-----------------------------------------------------------------')

# import feature_column
from tensorflow import feature_column

# Define feature columns for bedrooms and bathrooms
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")

# Define the feature columns
feature_list = [bedrooms, bathrooms]


def input_fn():
    # Define the labels
    labels = np.array(housing['price'], np.float32)
    # Define the features
    features = {'bedrooms': np.array(housing['bedrooms'], np.float32),
                'bathrooms': np.array(housing['bathrooms'], np.float32)}
    return features, labels


# 35. Defining Estimators
print("\n35. Defining Estimators")
print('-----------------------------------------------------------------')

from tensorflow.estimator import DNNRegressor

# Define the model and set the number of steps
model = DNNRegressor(feature_columns=feature_list, hidden_units=[2, 2])
model.train(input_fn, steps=1)
