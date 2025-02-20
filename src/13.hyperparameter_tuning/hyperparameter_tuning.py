# Hyperparameter tuning in Python
# Load libraries
from itertools import product

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
from sklearn.model_selection import train_test_split, cross_val_score

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
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

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

# 7. Build Grid Search Functions
print("--------------------------------------------")
print('7. Build Grid Search Functions')
print("--------------------------------------------")


# Create the function
def gbm_grid_search(learn_rate, max_depth):
    # Create the model
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth)

    # Use the model to make predictions
    predictions = model.fit(X_train, y_train).predict(X_test)

    # Return the hyperparameters and score
    return ([learn_rate, max_depth, accuracy_score(y_test, predictions)])


# 8. Iteratively tune multiple hyperparameters
print("--------------------------------------------")
print('8. Iteratively tune multiple hyperparameters')
print("--------------------------------------------")

# Create a list of values for each hyperparameter
results_list = []
learn_rate_list = [0.01, 0.1, 0.5]
max_depth_list = [2, 4, 6]

# Create the for loop
for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        results_list.append(gbm_grid_search(learn_rate, max_depth))

# Print results
print(results_list)


# Extend the function to iunclude the subsample parameter
def gbm_grid_search_extended(learn_rate, max_depth, subsample):
    # Create the model
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth, subsample=subsample)

    # Use the model to make predictions
    predictions = model.fit(X_train, y_train).predict(X_test)

    # Return the hyperparameters and score
    return ([learn_rate, max_depth, subsample, accuracy_score(y_test, predictions)])


results_list = []

# Create the new list to test
subsample_list = [0.4, 0.6]

for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        for subsample in subsample_list:
            results_list.append(gbm_grid_search_extended(learn_rate, max_depth, subsample))

# Print results
print(results_list)

# 9. GridSearchCV with Scikit Learn
print("--------------------------------------------")
print('9. GridSearchCV with Scikit Learn')
print("--------------------------------------------")

# Load libraries
from sklearn.model_selection import GridSearchCV

# Create a RandomForestClassifier
rf_class = RandomForestClassifier(criterion='entropy')

# Create a dictionary of hyperparameters to search
param_grid = {'max_depth': [2, 4, 8, 15], 'max_features': ['auto', 'sqrt']}

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(
    estimator=rf_class,
    param_grid=param_grid,
    scoring='roc_auc',  # Ensure this is the correct scoring method
    n_jobs=4,  # Consider using a specific number of jobs
    cv=5,  # Ensure the correct number of cross-validation folds
    refit=True,
    return_train_score=True)

print(grid_rf_class)

# 10. Exploring the grid search results
print("--------------------------------------------")
print('10. Exploring the grid search results')
print("--------------------------------------------")

# Fit the grid search object to the data
grid_rf_class.fit(X_train, y_train)

# Get the results of the grid search
grid_rf_class.predict(X_test)

# Read the cv_results property into a DataFrame & print it out
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
print(cv_results_df)

# Extract and print the column with a dictionary of hyperparameters used
column = cv_results_df.loc[:, ['params']]
print(column)

# Extract and print the row that had the best mean test score
best_row = cv_results_df[cv_results_df['rank_test_score'] == 1]
print(best_row)

# 11. Analyzing the best results
print("--------------------------------------------")
print('11. Analyzing the best results')
print("--------------------------------------------")

# Print out the ROC_AUC score from the best-performing square
best_score = grid_rf_class.best_score_
print(best_score)

# Create a variable from the row related to the best-performing square
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
best_row = cv_results_df.loc[[grid_rf_class.best_index_]]
print(best_row)

# Get the n_estimators parameter from the best-performing square and print
best_n_estimators = grid_rf_class.best_estimator_.n_estimators
print(best_n_estimators)

# 12. Using the best results
print("--------------------------------------------")
print('12. Using the best results')
print("--------------------------------------------")

# See what type of object the best_estimator_ property is
print(type(grid_rf_class.best_estimator_))

# Create an array of predictions directly using the best_estimator_ property
predictions = grid_rf_class.best_estimator_.predict(X_test)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# Now create a confusion matrix
print('Confusion Matrix: \n', confusion_matrix(y_test, predictions))

# Get the ROC-AUC score
predictions_proba = grid_rf_class.best_estimator_.predict_proba(X_test)[:, 1]
print('ROC-AUC Score: \n', roc_auc_score(y_test, predictions_proba))

# 13. Randomly Sample Hyperparameters
print("--------------------------------------------")
print('13. Randomly Sample Hyperparameters')
print("--------------------------------------------")

# Load libraries

# Create a list of values for the learning_rate hyperparameter
learn_rate_list = list(np.linspace(0.01, 1.5, 200))

# Create a list of values for the min_samples_leaf hyperparameter
min_samples_list = list(range(10, 41))

# Combination list
combinations_list = [list(x) for x in product(learn_rate_list, min_samples_list)]

# Sample hyperparameter combinations for a random search.
random_combinations_index = np.random.choice(range(0, len(combinations_list)), 250, replace=False)
combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]

# Print the result
print(combinations_random_chosen)

# 14. Randomly Search with Random Forest
print("--------------------------------------------")
print('14. Randomly Search with Random Forest')
print("--------------------------------------------")

# Create lists for criterion and max_features
criterion_list = ['gini', 'entropy']
max_feature_list = ['auto', 'sqrt', 'log2', None]

# Create a list of values for the max_depth hyperparameter
max_depth_list = list(range(3, 56))

# Combination list
combinations_list = [list(x) for x in product(criterion_list, max_feature_list, max_depth_list)]

# Sample hyperparameter combinations for a random search.
combinations_random_chosen = np.random.choice(range(0, len(combinations_list)), 150)

# Print the result
print(combinations_random_chosen)

# 15. Visualizing a Random Search
print("--------------------------------------------")
print('15. Visualizing a Random Search')
print("--------------------------------------------")

# Load libraries
import matplotlib.pyplot as plt


def sample_and_visualize_hyperparameters(n_samples):
    # If asking for all combinations, just return the entire list.
    if n_samples == len(combinations_list):
        combinations_randomly_chosen = combinations_list
    else:
        combinations_randomly_chosen = []
        random_combinations_indexed = np.random.choice(range(0, len(combinations_list)), n_samples, replace=False)
        combinations_randomly_chosen = [combinations_list[x] for x in random_combinations_indexed]

    # Pull out the X and Y to plot
    rand_y, rand_x = [x[0] for x in combinations_randomly_chosen], [x[1] for x in combinations_randomly_chosen]

    # Plot
    plt.clf()
    plt.scatter(rand_y, rand_x, c=['b'] * len(combinations_randomly_chosen))
    plt.gca().set(xlabel='learn_rate', ylabel='min_samples_leaf', title='Random Search Hyperparameters')

    # plt.gca().set_xlim(x_lims)
    # plt.gca().set_ylim(y_lims)
    plt.show()


# Confirm how many hyperparameter combinations & print

# Importing this created file since the datacamp environment
# variable has a totally different count of combinations between the exercises and this can cause confusion

combinations_list = pd.read_csv('../../data/13.hyperparameter_tuning/random_search_combo_list.csv', header=0,
                                usecols=[1, 2])

combinations_list = combinations_list.values.tolist()
number_combs = len(combinations_list)
print(number_combs)

# Sample and visualise specified hyperparameter combinations (50, 500 and 1500 combinations)
for x in [50, 500, 1500]:
    sample_and_visualize_hyperparameters(x)

# Sample all the hyperparameter combinations & visualise
sample_and_visualize_hyperparameters(number_combs)

# 16. Random Search with Scikit-learn - The RandomizedSearchCV Object
print("--------------------------------------------")
print('16. Random Search with Scikit-learn - The RandomizedSearchCV Object')
print("--------------------------------------------")

# Load libraries
from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid
param_grid = {'learning_rate': np.linspace(0.1, 2, 150), 'min_samples_leaf': list(range(20, 65))}

# Create a random search object
random_GBM_class = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(),
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy',  # Ensure this is 'accuracy' as per the instructions
    n_jobs=4,
    cv=5,
    random_state=1,
    refit=True,
    return_train_score=True
)

# Fit to the training data
random_GBM_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_GBM_class.cv_results_['param_learning_rate'])
print(random_GBM_class.cv_results_['param_min_samples_leaf'])

# 17. RandomSearchCV in Scikit-learn
print("--------------------------------------------")
print('17. RandomSearchCV in Scikit-learn')
print("--------------------------------------------")

# Create the parameter grid
param_grid = {"max_depth": list(range(5, 26)), "max_features": ['auto', 'sqrt']}

# Create a random search object
random_rf_class = RandomizedSearchCV(
    estimator=RandomForestClassifier(n_estimators=80),
    param_distributions=param_grid,
    n_iter=5,
    scoring='roc_auc',
    n_jobs=4,
    cv=3,
    random_state=1,
    refit=True,
    return_train_score=True
)

# Fit to the training data
random_rf_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_rf_class.cv_results_['param_max_depth'])
print(random_rf_class.cv_results_['param_max_features'])

# 18. Grid and Random Search Side by Side
print("--------------------------------------------")
print('18. Grid and Random Search Side by Side')
print("--------------------------------------------")


def visualize_search(grid_combinations_chosen, random_combinations_chosen):
    grid_y, grid_x = [x[0] for x in grid_combinations_chosen], [x[1] for x in grid_combinations_chosen]
    rand_y, rand_x = [x[0] for x in random_combinations_chosen], [x[1] for x in random_combinations_chosen]

    # Plot all together
    plt.scatter(grid_y + rand_y, grid_x + rand_x, c=['red'] * 300 + ['blue'] * 300)
    plt.gca().set(xlabel='learn_rate', ylabel='min_samples_leaf', title='Grid and Random Search Hyperparameters')
    # plt.gca().set_xlim(x_lims)
    # plt.gca().set_ylim(y_lims)
    plt.show()


# Sample grid coordinates
grid_combinations_chosen = combinations_list[0:300]

# Print results
print(grid_combinations_chosen)

# Create a list of sample indexes
sample_indexes = list(range(0, len(combinations_list)))

# Randomly sample 300 indexes
random_indexes = np.random.choice(sample_indexes, 300, replace=False)

# Use indexes to create the random sample
random_combinations_chosen = [combinations_list[index] for index in random_indexes]

# Call the function to produce the visualization
visualize_search(grid_combinations_chosen, random_combinations_chosen)

# 19. Informed Search - Visualizing Coarse to Fine
print("--------------------------------------------")
print('19. Informed Search - Visualizing Coarse to Fine')
print("--------------------------------------------")

# Confirm the size of the combinations_list
number_combs = len(combinations_list)
print(number_combs)


def visualize_hyperparameter(name):
    plt.clf()
    plt.scatter(results_df[name], results_df['accuracy'], c=['blue'] * 500)
    plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
    plt.gca().set_ylim([0, 100])
    plt.show()


combinations_list = pd.read_csv('../../data/13.hyperparameter_tuning/informed_search_combo_list.csv', header=0,
                                usecols=[1, 2, 3])

combinations_list = combinations_list.values.tolist()

results_df = pd.read_csv('../../data/13.hyperparameter_tuning/informed_search_hyperparameter_accuracy_result.csv',
                         header=0, usecols=[1, 2, 3, 4])

# Sort the results_df by accuracy and print the top 10 rows
print(results_df.sort_values(by='accuracy', ascending=False).head(10))

# Confirm which hyperparameters were used in this search
print(results_df.columns)

# Call visualize_hyperparameter() with each hyperparameter in turn
visualize_hyperparameter('learn_rate')
visualize_hyperparameter('min_samples_leaf')
visualize_hyperparameter('max_depth')

# 20. Informed Search - Coarse to Fine Iterations
print("--------------------------------------------")
print('20. Informed Search - Coarse to Fine Iterations')
print("--------------------------------------------")


def visualize_first():
    for name in results_df.columns[0:2]:
        plt.clf()
        plt.scatter(results_df[name], results_df['accuracy'], c=['blue'] * 500)
        plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
        plt.gca().set_ylim([0, 100])
        x_line = 20
        if name == "learn_rate":
            x_line = 1
        plt.axvline(x=x_line, color="red", linewidth=4)
        plt.show()


visualize_first()

# Create some combinations lists & combine
max_depth_list = list(range(1, 21))
learn_rate_list = np.linspace(0.001, 1, 50)

results_df2 = pd.read_csv('../../data/13.hyperparameter_tuning/informed_search_hyperparameters_accuracy_2.csv',
                          header=0, usecols=[1, 2, 3])


def visualize_second():
    for name in results_df2.columns[0:2]:
        plt.clf()
        plt.scatter(results_df2[name], results_df2['accuracy'], c=['blue'] * 1000)
        plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
        plt.gca().set_ylim([0, 100])
        plt.show()


visualize_second()

# 21. Informed Search - Bayesian stats
print("--------------------------------------------")
print('21. Informed Search - Bayesian stats')
print("--------------------------------------------")

# Assign probabilities to variables
p_unhappy = 0.15
p_unhappy_close = 0.35

# Probabilities of being unhappy and close to the store
p_close = 0.07

# Probability unhappy person will close
p_close_unhappy = p_unhappy_close * p_close / p_unhappy
print(p_close_unhappy)

# 22. Informed Search - Bayesian Hyperparameter tuning with Hyperopt
print("--------------------------------------------")
print('22. Informed Search - Bayesian Hyperparameter tuning with Hyperopt')
print("--------------------------------------------")

# Load libraries
from hyperopt import hp, fmin, tpe

# Set up space dictionary with specified hyperparameters
space = {'max_depth': hp.quniform('max_depth', 2, 10, 2), 'learning_rate': hp.uniform('learning_rate', 0.001, 0.9)}


# Set up objective function
def objective(params):
    params = {'max_depth': int(params['max_depth']), 'learning_rate': params['learning_rate']}
    gbm_clf = GradientBoostingClassifier(n_estimators=100, **params)
    best_score = cross_val_score(gbm_clf, X_train, y_train, scoring='accuracy', cv=2, n_jobs=4).mean()
    loss = 1 - best_score
    return loss


# Run the algorithm
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, rstate=np.random.default_rng(42))
print(best)

# 23. Informed Search - Genetic Hyperparameter tuning with TPOT
print("--------------------------------------------")
print('23. Informed Search - Genetic Hyperparameter tuning with TPOT')
print("--------------------------------------------")

# Load libraries
from tpot import TPOTClassifier

# Assign the values outlined to the inputs
number_generations = 3
population_size = 4
offspring_size = 3
scoring_function = 'accuracy'

# Create a TPOT classifier
tpot_clf = TPOTClassifier(generations=number_generations, population_size=population_size,
                          offspring_size=offspring_size,
                          scoring=scoring_function, verbosity=2, random_state=42, cv=2)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))

# 24. Informed Search - Analyzing TPOT's stability
print("--------------------------------------------")
print('24. Informed Search - Analyzing TPOT stability')
print("--------------------------------------------")

# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=2, population_size=4, offspring_size=3, scoring='accuracy', cv=2,
                          verbosity=2, random_state=42)  # You can use other random states to get different results

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))
