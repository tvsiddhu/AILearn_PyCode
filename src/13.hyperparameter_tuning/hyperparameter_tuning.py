# Hyperparameter tuning in Python
# Load libraries
import pandas as pd
import pydotplus
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.tree import export_graphviz

# 1. Extracting a Logistic Regression model
print("--------------------------------------------")
print('1. Extracting a Logistic Regression model')
print("--------------------------------------------")
# Load data
credit_card_df = pd.read_csv('../../data/13.hyperparameter_tuning/credit-card-full.csv')

# Create feature matrix and target vector
X = credit_card_df.iloc[:, 1:-1]
y = credit_card_df['default payment next month']

# # fetch dataset
# default_of_credit_card_clients = fetch_ucirepo(id=350)
#
# # data (as pandas dataframes)
# X = default_of_credit_card_clients.data.features
# y = default_of_credit_card_clients.data.targets
#
# # metadata
# print(default_of_credit_card_clients.metadata)
#
# # variable information
# print(default_of_credit_card_clients.variables)

# Split data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create logistic regression
log_reg_clf = linear_model.LogisticRegression()

# # Create a list of original variable names from the training DataFrame
# original_variables = X_train.columns
#
# # Extract the coefficients of the logistic regression estimator
# model_coefficients = log_reg_clf.fit(X_train, y_train).coef_[0]
#
# # Create a DataFrame of the variables and coefficients & print it out
# coefficient_df = pd.DataFrame({'Variable': original_variables, 'Coefficient': model_coefficients})
# print(coefficient_df)
#
# # Print out the top 3 positive variables
# top_three_df = coefficient_df.sort_values(by='Coefficient', axis=0, ascending=False)[0:3]
# print(top_three_df)

log_reg_clf.fit(X_train, y_train)

# Get the original variable names
original_variables = list(X_train.columns)
# Zip together the names and coefficients
zipped_together = list(zip(original_variables, log_reg_clf.coef_[0]))
coefs = [list(x) for x in zipped_together]
# Put into a DataFrame with column labels
coefs = pd.DataFrame(coefs, columns=["Variable", "Coefficient"])

# 2. Extracting a Random Forest parameter
print("--------------------------------------------")
print('2. Extracting a Random Forest parameter')
print("--------------------------------------------")

# Load libraries
from sklearn.ensemble import RandomForestClassifier

# Create random forest classifier
rf_clf = RandomForestClassifier(max_depth=2)

# Extract the feature importances
rf_clf.fit(X_train, y_train)

# Extract the 7th (index 6) tree from the random forest
chosen_tree = rf_clf.estimators_[6]

# Export the image to a dot file
export_graphviz(chosen_tree, out_file='../../res/img/tree.dot', feature_names=original_variables, rounded=True,
                precision=1)

# Use dot file to create a graph
graph = pydotplus.graph_from_dot_file('../../res/img/tree.dot')

# Write graph to a png file
graph.write_png('../../res/img/tree_viz_image.png')

# Display the graph
imgplot = plt.imshow(plt.imread('../../res/img/tree_viz_image.png'))
plt.show()

# Extract the parameters and level of the top (index 0) node
split_column = chosen_tree.tree_.feature[0]
split_column_name = original_variables[split_column]
split_value = chosen_tree.tree_.threshold[0]

# Print out the feature and level
print(f'The top node is split on feature {split_column_name} at a value of {split_value}')

# 3. Exploring Random Forest Hyperparameters
print("--------------------------------------------")
print('3. Exploring Random Forest Hyperparameters')
print("--------------------------------------------")

rf_clf_old = rf_clf
print('Random Forest Hyperparameters:')
print(rf_clf_old.get_params())

# Get confusion matrix & accuracy for the rf_clf_old model
from sklearn.metrics import confusion_matrix, accuracy_score

# Predict the test set
rf_old_predictions = rf_clf_old.predict(X_test)

# Print the confusion matrix
print("No Hyper-parameters Set: \nConfusion Matrix: \n {} \n Accuracy Score: \n {}".format(
    confusion_matrix(y_test, rf_old_predictions),
    accuracy_score(y_test, rf_old_predictions)))

# Create a new random forest classifier with better hyperparameters
rf_clf_new = RandomForestClassifier(n_estimators=500)

# Fit the model to the data and obtain predictions
rf_new_predictions = rf_clf_new.fit(X_train, y_train).predict(X_test)

# Print the confusion matrix
print("With n_estimators Hyper-parameter set tp 500: \nConfusion Matrix: \n {} \n Accuracy Score: \n {}".format(
    confusion_matrix(y_test, rf_new_predictions),
    accuracy_score(y_test, rf_new_predictions)))

# 4. Hyperparameters of KNN
print("--------------------------------------------")
print('4. Hyperparameters of KNN')
print("--------------------------------------------")

# Load libraries
from sklearn.neighbors import KNeighborsClassifier

# Build a knn estimator for eavh value of n_neighbors
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_20 = KNeighborsClassifier(n_neighbors=20)

# Fit each of the training data & produce predictions
knn_5_predictions = knn_5.fit(X_train, y_train).predict(X_test)
knn_10_predictions = knn_10.fit(X_train, y_train).predict(X_test)
knn_20_predictions = knn_20.fit(X_train, y_train).predict(X_test)

# Get an accuracy score for each of the models
knn_5_accuracy = accuracy_score(y_test, knn_5_predictions)
knn_10_accuracy = accuracy_score(y_test, knn_10_predictions)
knn_20_accuracy = accuracy_score(y_test, knn_20_predictions)

# Print out the results
print(f'KNN 5 Accuracy: {knn_5_accuracy}')
print(f'KNN 10 Accuracy: {knn_10_accuracy}')
print(f'KNN 20 Accuracy: {knn_20_accuracy}')

# 5. Automating Hyperparameter Choice
print("--------------------------------------------")
print('5. Automating Hyperparameter Choice')
print("--------------------------------------------")
# Import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting Classifier
gbc = GradientBoostingClassifier()

# Create a dictionary of hyperparameters to search
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
results_list = []

# Create the for loop to evaluate model predictions for each value of learning_rates
for learning_rate in learning_rates:
    gbc.set_params(learning_rate=learning_rate)
    predictions = gbc.fit(X_train, y_train).predict(X_test)
    # Save the learning rate and accuracy score
    results_list.append((learning_rate, accuracy_score(y_test, predictions)))

# Gather everything into a DataFrame
results_df = pd.DataFrame(results_list, columns=['learning_rate', 'accuracy'])
print(results_df)

# 6. Building Learning Curves
print("--------------------------------------------")
print('6. Building Learning Curves')
print("--------------------------------------------")

# Load libraries
import numpy as np

# Set the learning rates & accuracies list
learn_rates = np.linspace(0.01, 2, num=30)
accuracies = []

# Create the for loop
for learn_rate in learn_rates:
    gbc.set_params(learning_rate=learn_rate)
    predictions = gbc.fit(X_train, y_train).predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

# Plot results
plt.plot(learn_rates, accuracies)
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy for different learning rates')
plt.show()
