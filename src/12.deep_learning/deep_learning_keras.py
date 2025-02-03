# Disable automatic date conversion globally
import warnings

import keras
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.units as munits
import numpy as np
import pandas as pd
import seaborn as sns
# from scikeras.wrappers import KerasClassifier
# Import the data from sklearn
from sklearn import datasets
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. Hello Nets
print("Hello Nets")
print("--------------------------")

munits.registry.clear()

# from tensorflow.python.keras.layers import Dense
# Import the Sequential model and Dense layer
# from tensorflow.python.keras.models import Sequential

# Create a Sequential model
model = keras.Sequential()

# Add an input layer and a hidden layer with 10 neurons
model.add(keras.layers.Dense(10, input_shape=(2,), activation="relu"))

# Add a 1-neuron output layer
model.add(keras.layers.Dense(1))

# Summarize your model
model.summary()

# 2. Counting Parameters
print("Counting Parameters")
print("--------------------------")

# Instantiate a new Sequential model
model = keras.Sequential()

# Add a Dense layer with five neurons and three inputs
model.add(keras.layers.Dense(5, input_shape=(3,), activation="relu"))

# Add a final Dense layer with one neuron and no activation function
model.add(keras.layers.Dense(1))

# Summarize your model
model.summary()

# 3. Build as shown
print("Build as shown")
print("--------------------------")

# Instantiate a Sequential model
model = keras.Sequential()

# Build the input and hidden layer
model.add(keras.layers.Dense(3, input_shape=(2,), activation="relu"))

# Add the output layer
model.add(keras.layers.Dense(1))

# Summarize your model
model.summary()

# 4. Specifying a model to predict orbit of a meteor
print("Specifying a model to predict orbit of a meteor")
print("--------------------------")

# Import the data

time_steps = pd.read_csv("../../data/12.deep_learning/meteor_time_steps.csv", header=None)  # features
time_steps = time_steps.iloc[:, 1].to_numpy()

y_positions = pd.read_csv("../../data/12.deep_learning/meteor_y_positions.csv", header=None)  # labels
y_positions = y_positions.iloc[:, 1].to_numpy()

# Instantiate a Sequential model
model = keras.Sequential()

# Add a Dense layer with 50 neurons and an input of 1 neuron
model.add(keras.layers.Dense(50, input_shape=(1,), activation="relu"))

# Add two Dense layers with 50 neurons and relu activation
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(50, activation="relu"))

# Emulate your model with a Dense layer and no activation
model.add(keras.layers.Dense(1))

# Compile your model
model.compile(optimizer='adam', loss='mse')

print("Training started..., this may take a while:")

# Fit your model on your data for 30 epochs
model.fit(time_steps, y_positions, epochs=30)

# Evaluate your model
print("Final loss value:", model.evaluate(time_steps, y_positions))

# 5. Predicting the orbit!
print("Predicting the orbit!")
print("--------------------------")


def plot_orbit(model_preds):
    axeslim = int(len(model_preds) / 2)
    """
    This function takes a model and uses it to plot the predicted trajectory of a meteor in the y-x plane."""

    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(np.arange(-axeslim, axeslim + 1), np.arange(-axeslim, axeslim + 1) ** 2, color="mediumslateblue",
            label="Scientist's Orbit")
    ax.plot(np.arange(-axeslim, axeslim + 1), model_preds, color="orange", label="Your orbit")

    # Set axis limits using set methods
    ax.set_xlim(-40, 41)
    ax.set_ylim(-5, 550)

    # Use set methods for title and legend
    ax.set_title("Predicted orbit vs Scientist's Orbit")
    ax.legend(loc="lower left")

    # Show the plot
    plt.show()


# Predict the twenty minutes orbit
twenty_min_orbit = model.predict(np.arange(-10, 11))

# Plot the twenty minutes orbit
plot_orbit(twenty_min_orbit)

# Predict the eighty-minute orbit
eighty_min_orbit = model.predict(np.arange(-40, 41))

# Plot the eighty-minute orbit
plot_orbit(eighty_min_orbit)

# 6. Exploring dollar bills
print("Exploring dollar bills")
print("--------------------------")

# Import the data
banknotes = pd.read_csv("../../data/12.deep_learning/banknote_authentication.csv", header=0)

# Use pairplot and set the hue to be our class
sns.pairplot(banknotes, hue='class')

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', banknotes.describe())

# Count the number of observations per class
print('Observations per class: \n', banknotes['class'].value_counts())

# 7. A binary classification model
print("A binary classification model")
print("--------------------------")

# Instantiate a Sequential model
banknotes_model = keras.Sequential()

# Add a Dense layer with 1 neuron and an input of 4 neurons with signmoid activation
banknotes_model.add(keras.layers.Dense(1, input_shape=(4,), activation="sigmoid"))

# Compile your model
banknotes_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Display a summary of your model
banknotes_model.summary()

# 8. Is the banknote fake or authentic?
print("Is the banknote fake or authentic?")
print("--------------------------")

# Split the data into train and test
from sklearn.model_selection import train_test_split

# Import the data
banknotes_dataset = pd.read_csv("../../data/12.deep_learning/banknotes.csv", header=0)

banknotes_X = banknotes_dataset.iloc[:, 0:4].apply(pd.to_numeric, errors='coerce').fillna(0)
banknotes_y = banknotes_dataset.iloc[:, 4].apply(pd.to_numeric, errors='coerce').fillna(0)

banknotes_X_train, banknotes_X_test, banknotes_y_train, banknotes_y_test = train_test_split(banknotes_X, banknotes_y,
                                                                                            test_size=0.30)

# Train your model for 20 epochs
banknotes_model.fit(banknotes_X_train, banknotes_y_train, epochs=20)

# Evaluate your model accuracy on the test set
accuracy = banknotes_model.evaluate(banknotes_X_test, banknotes_y_test)[1]

# Print accuracy
print('Accuracy:', accuracy)

# 9. Multi-label classification
print("Multi-label classification")
print("--------------------------")

# Instantiate a Sequential model
model = keras.Sequential()

# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(keras.layers.Dense(128, input_shape=(4,), activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))

# Add a dense layer with as many neurons as competitors
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile your model using categorical_crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 10. Preparing your input data
print("Preparing your input data")
print("--------------------------")

# Import the darts dataset
darts = pd.read_csv("../../data/12.deep_learning/darts.csv", header=0)

# Transform the target column to categorical
# darts['competitor'] = darts['competitor'].astype('category')

# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes

# Print the label encoded competitors
print('Label encoded competitors: \n', darts.competitor.head())

# Use to_categorical on your labels
coordinates = darts.drop(['competitor'], axis=1)
competitors = keras.utils.to_categorical(darts.competitor)

# Now print the one-hot encoded labels
print('One-hot encoded competitors: \n', competitors)

# 11. Training on dart throwers
print("Training on dart throwers")
print("--------------------------")

coordinates = darts.drop(['competitor'], axis=1)
competitors = keras.utils.to_categorical(darts.competitor)

# Ensure the input shape matches the data shape
model = keras.Sequential()
model.add(keras.layers.Dense(128, input_shape=(2,), activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile your model using categorical_crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Split the data into train and test
coord_train, coord_test, competitors_train, competitors_test = train_test_split(coordinates, competitors,
                                                                                test_size=0.25)

# Train your model on the training data for 200 epochs
model.fit(coord_train, competitors_train, epochs=200)

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(coord_test, competitors_test)[1]

# Print accuracy
print('Accuracy:', accuracy)

# 12. Softmax predictions
print("Softmax predictions")
print("--------------------------")

# Test throw and competitors dataset (small)
# coords_small_test = np.array([
#     [337, 0.209, -0.077],
#     [295, 0.082, -0.721],
#     [243, 0.198, -0.675],
#     [91, -0.349, 0.035],
#     [375, 0.215, 0.184]
# ])
coords_small_test = np.array([
    [337, 0.209],
    [295, 0.082],
    [243, 0.198],
    [91, -0.349],
    [375, 0.215]
])

competitors_small_test = np.array([
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

# Predict on coords_small_test

preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions', 'True labels'))
for i, pred in enumerate(preds):
    print("{} | {}".format(pred, competitors_small_test[i]))

# Extract the position of highest probability from each pred vector
preds_chosen = [np.argmax(pred) for pred in preds]

# Print the most likely competitor for each throw
print(preds_chosen)

# 13. An irrigation machine
print("An irrigation machine")
print("--------------------------")

# Import the data
irrigation = pd.read_csv("../../data/12.deep_learning/irrigation_machine.csv", header=0)
irrigation = irrigation.drop('Unnamed: 0', axis=1)

sensors = irrigation.drop(['parcel_0', 'parcel_1', 'parcel_2'], axis=1)
parcels = irrigation[['parcel_0', 'parcel_1', 'parcel_2']]

print("Sensor data:", sensors.head())
print("Parcel data:", parcels.head())

# Instantiate a Sequential model
model = keras.Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron input
model.add(keras.layers.Dense(64, input_shape=(20,), activation='relu'))

# Add an output layer of 3 neurons with sigmoid activation
model.add(keras.layers.Dense(3, activation='sigmoid'))

# Compile your model with adam and binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display a summary of your model
model.summary()

sensors_train = sensors.iloc[:1000]
sensors_test = sensors.iloc[1000:]
parcel_train = parcels.iloc[:1000]
parcel_test = parcels.iloc[1000:]

sensors_train, sensors_test, parcels_train, parcels_test = train_test_split(sensors, parcels, test_size=0.3)

# Train your model for 100 epochs using sensor_train and parcel_train and validation split 0.2
model.fit(sensors_train, parcels_train, epochs=100, validation_split=0.2)

# Predict on sensor_test and assign to preds with round 2 decimal places
preds = model.predict(sensors_test)
preds_rounded = np.round(preds, 2)

# Print preds vs true values
print("{:30} | {}".format('Raw Model Predictions', 'True labels'))
for i, pred in enumerate(preds_rounded):
    print("{} | {}".format(pred, parcels_test.iloc[i].values))

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(sensors_test, parcels_test)[1]

# Print accuracy
print('Accuracy:', accuracy)

# 14. The history callback
print("The history callback")
print("--------------------------")

X = sensors
y = parcels

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create the model
model = keras.Sequential()
model.add(keras.layers.Dense(64, input_shape=(20,), activation='relu'))
model.add(keras.layers.Dense(3, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and store the training object
h_callback = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))


def plot_loss(loss, val_loss):
    """
    This function plots the training and validation loss"""

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot loss values
    ax.plot(loss, label="Train")
    ax.plot(val_loss, label="Test")

    # Set labels and title
    ax.set_title("Model loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    # Add legend
    ax.legend(loc="upper right")

    # Show the plot
    plt.show()


def plot_accuracy(acc, val_acc):
    """

    :param acc:
    :param val_acc:
    :return:
    """

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot accuracy values and ensure x-axis uses numeric values
    ax.plot(np.arange(len(acc)), acc, label="Train")
    ax.plot(np.arange(len(val_acc)), val_acc, label="Test")

    plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))

    # Set labels and title
    ax.set_title("Model Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    # Add legend
    ax.legend(loc="upper left")

    # Show the plot
    plt.show()


# Plot the training loss
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# Plot the training accuracy
plot_accuracy(h_callback.history['accuracy'], h_callback.history['val_accuracy'])

# 15. Early stopping your model
print("Early stopping your model")
print("--------------------------")

banknotes = pd.read_csv("../../data/12.deep_learning/banknotes.csv", header=0)

X = banknotes.iloc[:, 0:4].apply(pd.to_numeric, errors='coerce').fillna(0)
y = banknotes.iloc[:, 4].apply(pd.to_numeric, errors='coerce').fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a Sequential model
model = keras.Sequential()

# Add a Dense layer with 12 neurons and an input of 4 neurons
model.add(keras.layers.Dense(12, input_shape=(4,), activation='relu'))

# Add a Dense layer with 8 neurons
model.add(keras.layers.Dense(8, activation='relu'))

# Add a final Dense layer with 1 neuron
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile your model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a callback to monitor val_acc
monitor_val_acc = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

# Train your model using the early stopping callback
model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), callbacks=[monitor_val_acc])

# 16. A combination of callbacks
print("A combination of callbacks")
print("--------------------------")

X = sensors
y = parcels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a Sequential model
model = keras.Sequential()

# Add a Dense layer with 64 neurons and an input of 20 neurons
model.add(keras.layers.Dense(64, input_shape=(20,), activation='relu'))

# Add two Dense layers with 30 neurons
model.add(keras.layers.Dense(30, activation='relu'))
model.add(keras.layers.Dense(30, activation='relu'))

# Add a final Dense layer with 3 neurons and sigmoid activation
model.add(keras.layers.Dense(3, activation='sigmoid'))

# Compile your model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stop on validation accuracy
monitor_val_acc = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

# Save the best model as best_banknote_model.h5
modelCheckpoint = keras.callbacks.ModelCheckpoint('best_banknote_model.h5', save_best_only=True)

# Fit your model passing in the callbacks
model.fit(X_train, y_train, epochs=1000000, validation_data=(X_test, y_test),
          callbacks=[monitor_val_acc, modelCheckpoint])

# 17. Learning the digits
print("Learning the digits")
print("--------------------------")

digits = datasets.load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a Sequential model
model = keras.Sequential()

# Add a Dense layer with 16 neurons with relu activation and an input shape of (64,)
model.add(keras.layers.Dense(16, input_shape=(64,), activation='relu'))

# Add a Dense layer with 10 output neurons and softmax activation
model.add(keras.layers.Dense(10, activation='softmax'))

# Compile your model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Test if your model is well assembled by predicting before training
preds = model.predict(X_test)

# Print preds
print('Predictions:', preds)

# 18. Is the model overfitting?
print("Is the model overfitting?")
print("--------------------------")

# Train your model for 60 epochs, using X_test and y_test as validation data
h_callback = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), verbose=0)

# Extract from the h_callback object loss and val_loss to plot the learning curve
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# 19. Do we need more data?
print("Do we need more data?")
print("--------------------------")

# Store initial model weights
initial_weights = model.get_weights()

# List for storing accuracies
train_accs = []
test_accs = []
training_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900]

# Train the model with different batch sizes
for size in training_sizes:
    # Get a fraction of data (training_size out of X_train)
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]

    # Reset the model to the initial weights and train it on the new data
    model.set_weights(initial_weights)
    model.fit(X_train_frac, y_train_frac, epochs=50, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor='loss')])

    # Evaluate and store both: the training data and the test data
    train_accs.append(model.evaluate(X_train, y_train)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])

# Plot train vs test accuracies
plt.plot(training_sizes, train_accs, 'o-', label='Training Accuracy')
plt.plot(training_sizes, test_accs, 'o-', label='Test Accuracy')
plt.legend()
plt.show()

# 20. Comparing activation functions
print("Comparing activation functions")
print("--------------------------")


def get_model(act_function):
    """
    This function returns a compiled neural network model of a certain activation function
    """

    # Create a Sequential model
    model = keras.Sequential()

    # Add a Dense layer with 64 neurons and relu activation
    model.add(keras.layers.Dense(64, input_shape=(64,), activation=act_function))

    # Add a Dense layer with 10 output neurons and softmax activation
    model.add(keras.layers.Dense(10, activation='softmax'))

    # Compile your model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# Activation functions to try
activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh']

# Loop over the activation functions
activation_results = {}

for act in activations:
    # Get a new model with the current activation
    model = get_model(act)

    # Fit the model
    h_callback = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)

    # Store the history callback results
    activation_results[act] = h_callback

    # Plot the learning curves
    plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

    # Print the accuracy
    print("Accuracy with", act, ":", model.evaluate(X_test, y_test)[1])

# 21. Comparing activation functions II
print("Comparing activation functions II")
print("--------------------------")

val_loss_per_function = {}
val_acc_per_function = {}

for k, v in activation_results.items():
    val_loss_per_function[k] = v.history['val_loss']
    val_acc_per_function[k] = v.history['val_accuracy']

# Create a dataframe from val_loss_per_function
val_loss = pd.DataFrame(val_loss_per_function)

# Call plot on the dataframe
val_loss.plot()
plt.show()

# Create a dataframe from val_acc_per_function
val_acc = pd.DataFrame(val_acc_per_function)

# Call plot on the dataframe
val_acc.plot()
plt.show()

# 22. Changing batch sizes
print("Changing batch sizes")
print("--------------------------")

# Create a Sequential model
model = get_model('relu')

# Fit your model for 5 epochs with a batch size of 1
model.fit(X_train, y_train, epochs=5, batch_size=1)

print("Accuracy with batch size of 1:", model.evaluate(X_test, y_test)[1])

# Fit your model for 5 epochs with a batch size of the training set
model.fit(X_train, y_train, epochs=5, batch_size=len(X_train))

print("Accuracy with batch size of training set:", model.evaluate(X_test, y_test)[1])

# 23. Batch normalizing a familiar model
print("Batch normalizing a familiar model")
print("--------------------------")

# Build your deep network
batchnorm_model = keras.Sequential()
batchnorm_model.add(keras.layers.Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(keras.layers.BatchNormalization())
batchnorm_model.add(keras.layers.Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(keras.layers.BatchNormalization())
batchnorm_model.add(keras.layers.Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(keras.layers.BatchNormalization())
batchnorm_model.add(keras.layers.Dense(10, activation='softmax', kernel_initializer='normal'))

# Compile your model with sgd
batchnorm_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Batch normalization model summary:")
batchnorm_model.summary()

# 24. Batch normalization effects
print("Batch normalization effects")
print("--------------------------")

standard_model = get_model('relu')

# Train your standard model, storing its history callback
standard_model = keras.Sequential()
standard_model.add(keras.layers.Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
standard_model.add(keras.layers.Dense(50, activation='relu', kernel_initializer='normal'))
standard_model.add(keras.layers.Dense(50, activation='relu', kernel_initializer='normal'))
standard_model.add(keras.layers.Dense(10, activation='softmax', kernel_initializer='normal'))

standard_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Standard model summary:", standard_model.summary())

# Train your standard model, storing its history callback
h1_callback = standard_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)

# Train the batch normalized model you recently built, store its history callback
h2_callback = batchnorm_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)


def compare_histories_acc(h1, h2):
    """
    This function compares two Keras model histories
    """

    plt.plot(h1.history['accuracy'], 'b', h1.history['val_accuracy'], 'r')
    plt.plot(h2.history['accuracy'], 'g', h2.history['val_accuracy'], 'm')
    plt.title('Batch Normalization Effects')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test', 'Train with Batch Normalization', 'Test with Batch Normalization'])
    plt.show()


# call compare_histories_acc passing in both model histories
compare_histories_acc(h1_callback, h2_callback)

# 25. Preparing a model for tuning
print("Preparing a model for tuning")
print("--------------------------")

# Import the data
# Removing this for now
# cancer_dataset = load_breast_cancer(return_X_y=True)
cancer = load_breast_cancer(as_frame=True)['frame']
X = cancer.drop('target', axis=1)
y = cancer['target']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Creates a model given an activation and learning rate
def create_model(learning_rate, activation):
    # Create an Adam optimizer with the given learning rate
    opt = keras.optimizers.Adam(learning_rate)

    # Create your binary classification model
    bin_model = keras.Sequential()
    bin_model.add(keras.layers.Dense(128, input_shape=(30,), activation=activation))
    bin_model.add(keras.layers.Dense(256, activation=activation))
    bin_model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Compile your model with your optimizer, loss, and metrics
    bin_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return bin_model


# 26. Tuning the model parameters
print("Tuning the model parameters")
print("--------------------------")

# Import KerasClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold  # , cross_val_score

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model)

# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256], 'epochs': [50, 100, 200],
          'learning_rate': [0.1, 0.01, 0.001]}

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions=params, cv=KFold(3), n_iter=5)

# Running random_search.fit
# random_search.fit(X_train, y_train)

# 27. Training with cross-validation
print("Training with cross-validation")
print("--------------------------")

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model(learning_rate=0.001, activation='relu'), epochs=50, batch_size=128,
                        verbose=0, model_kwargs={})

# Commenting the below lines out due to incompatability between scikeras and sklearn cross_val_score and I don't want
# to downgrade scikit-learn to fix this

# # Calculate the accuracy score for each fold
# kfolds = cross_val_score(model, X_train, y_train, cv=3)
#
# # Print the mean accuracy
# print('The mean accuracy was:', kfolds.mean())
#
# # Print the accuracy standard deviation
# print('With a standard deviation of:', kfolds.std())

# 28. It's a flow of tensors
print("It's a flow of tensors")
print("--------------------------")

# Import the data
print("banknotes_X_test:", banknotes_X_test)
print("banknotes_y_test:", banknotes_y_test)

# Instantiate a Sequential model
banknotes_model = keras.Sequential()

# Add a Dense layer with 1 neuron and an input of 4 neurons with signmoid activation
banknotes_model.add(keras.layers.Dense(2, input_shape=(4,), activation="sigmoid"))

# Add an output layer
banknotes_model.add(keras.layers.Dense(1))

# Compile your model
banknotes_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Display a summary of your model
banknotes_model.summary()

# Fit the model
banknotes_model.fit(banknotes_X_train, banknotes_y_train, epochs=20)

# Input tensor from the 1st layer of the model
inp = banknotes_model.layers[0].input

# Output tensor from the 1st layer of the model
out = banknotes_model.layers[0].output

import tensorflow as tf


@tf.function
def inp_out(inp, out):
    return banknotes_model(inp)


# Convert KerasTensor to Tensor
inp_tensor = tf.convert_to_tensor(banknotes_X_test, dtype=tf.float32)
out_tensor = tf.convert_to_tensor(banknotes_y_test, dtype=tf.float32)
result = inp_out(inp_tensor, out_tensor)
print(result)

# 29. Neural separation
print("Neural separation")
print("--------------------------")

for i in range(0, 21):
    print("Epoch:", i)
    h = banknotes_model.fit(banknotes_X_train, banknotes_y_train, epochs=1, verbose=0, batch_size=16)
    if i % 4 == 0:
        print("Loss at epoch", i, ":", h.history['loss'][0])
        # Get the output of the first layer
        layer_output = banknotes_model.layers[0].output

    print("Accuracy:", banknotes_model.evaluate(banknotes_X_test, banknotes_y_test)[1])


def plot():
    fig, ax = plt.subplots()
    plt.scatter(layer_output[:, 0], layer_output[:, 1], c=banknotes_y_test, edgecolors='none')
    plt.title('Epoch: {}, Test Accuracy: {:3.1f} %'.format(i + 1, test_accuracy * 100.0))
    plt.show()


for i in range(0, 21):
    # Train model for 1 epoch
    h = banknotes_model.fit(banknotes_X_train, banknotes_y_train, batch_size=16, epochs=1, verbose=0)
    if i % 4 == 0:
        # Get the output of the first layer
        layer_output = banknotes_model.layers[0](banknotes_X_test)

        # Evaluate model accuracy for this epoch
        test_accuracy = banknotes_model.evaluate(banknotes_X_test, banknotes_y_test)[1]

        # Plot 1st vs 2nd neuron output
        plot()

# 30. Building an autoencoder
print("Building an autoencoder")
print("--------------------------")

# Import the MNIST dataset
emnist = keras.datasets.mnist
# Load the MNIST dataset
(mnist_X_train, mnist_y_train), (mnist_X_test, mnist_y_test) = emnist.load_data()

# Normalize the images
mnist_X_train = mnist_X_train / 255.0
mnist_X_test = mnist_X_test / 255.0

# Flatten the images
mnist_X_train = mnist_X_train.reshape(-1, 784)
mnist_X_test = mnist_X_test.reshape(-1, 784)

# Start with a sequential model
autoencoder = keras.Sequential()

# Add a dense layer with input the size of the image
autoencoder.add(keras.layers.Dense(32, input_shape=(784,), activation='relu'))

# Add an output layer with as many neurons as the image
autoencoder.add(keras.layers.Dense(784, activation='sigmoid'))

# Compile your model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Print a summary of the autoencoder model
autoencoder.summary()

# 31. De-noising like an autoencoder
print("De-noising like an autoencoder")
print("--------------------------")

mnist_X_test_noise = np.load("../../data/12.deep_learning/X_test_MNIST_noise.npy")
mnist_y_test_noise = np.load("../../data/12.deep_learning/y_test_MNIST.npy")


# -------------
def show_encodings(encoded_imgs, number=1):
    n = 5  # how many digits we will display
    original = mnist_X_test_noise
    original = original[np.where(mnist_y_test_noise == number)]
    encoded_imgs = encoded_imgs[np.where(mnist_y_test_noise == number)]
    plt.figure(figsize=(20, 4))
    # plt.title('Original '+str(number)+' vs Encoded representation')
    for i in range(min(n, len(original))):
        # display original imgs
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display encoded imgs
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(np.tile(encoded_imgs[i], (32, 1)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def compare_plot(original, decoded_imgs):
    n = 4  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.title('Noisy vs Decoded images')
    plt.show()


# Build your encoder by using the first lasyer of your autoencoder
encoder = keras.Sequential()
# encoder.add(keras.layers.Dense(32, input_shape=(784,), activation='relu'))
encoder.add(autoencoder.layers[0])

# Encode the noisy images and show the encodings for your favorite number [0-9]
# Ensure the input data is reshaped correctly

encodings = encoder.predict(mnist_X_test_noise)
show_encodings(encodings, number=1)

# 32. De-noising like an autoencoder II
print("De-noising like an autoencoder II")
print("--------------------------")

decoder_layer = autoencoder.layers[-1]
decoder = keras.models.Model()

# Predict on the noisy images with your autoencoder
decoded_imgs = autoencoder.predict(mnist_X_test_noise)

# Plot noisy vs decoded images
print("Noisy vs Decoded images")
compare_plot(mnist_X_test_noise, decoded_imgs)

# Predict on the noisy images with your autoencoder
decoded_imgs = autoencoder.predict(mnist_X_test_noise)

# Plot noisy vs decoded images
compare_plot(mnist_X_test_noise, decoded_imgs)

