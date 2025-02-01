# Sourcefile for deep learning with keras
# 1. Hello Nets
print("Hello Nets")
print("--------------------------")

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import keras
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

# Disable automatic date conversion globally
import matplotlib.units as munits
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

# Predict the eighty minute orbit
eighty_min_orbit = model.predict(np.arange(-40, 41))

# Plot the eighty minute orbit
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
model = keras.Sequential()

# Add a Dense layer with 1 neuron and an input of 4 neurons with signmoid activation
model.add(keras.layers.Dense(1, input_shape=(4,), activation="sigmoid"))

# Compile your model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Display a summary of your model
model.summary()

# 8. Is the banknote fake or authentic?
print("Is the banknote fake or authentic?")
print("--------------------------")

# Split the data into train and test
from sklearn.model_selection import train_test_split

# Import the data
banknotes_dataset = pd.read_csv("../../data/12.deep_learning/banknotes.csv", header=0)

X = banknotes_dataset.iloc[:, 0:4].apply(pd.to_numeric, errors='coerce').fillna(0)
y = banknotes_dataset.iloc[:, 4].apply(pd.to_numeric, errors='coerce').fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train your model for 20 epochs
model.fit(X_train, y_train, epochs=20)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test, y_test)[1]

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
model.fit(X_train, y_train, epochs=1000000, validation_data=(X_test, y_test), callbacks=[monitor_val_acc, modelCheckpoint])

