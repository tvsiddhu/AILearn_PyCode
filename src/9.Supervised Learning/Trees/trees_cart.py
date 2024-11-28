# Training your first classification Tree
# Load the breast cancer data (ignore the file path)
import matplotlib.pyplot as plt
import numpy as np
# ---- Code if you want to import from the file directly ----
# Import the Wisconsin Breast Cancer dataset
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# from sklearn.datasets import load_breast_cancer
# data = load_breast_cancer()
# # Create feature and target arrays
# X = data.data
# y = data.target

data = pd.read_csv('../../../data/supervised_learning/trees_breast_cancer_wisconsin.csv')

# Create feature and target arrays
X = data.drop('diagnosis', axis=1).iloc[:, :2].values
y = data['diagnosis'].values

# Replace 'M' with 1 and 'B' with 0
y = np.where(y == 'M', 1, 0)

# Replace NaN values with mean of that column
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Define SEED
SEED = 1

# Split into training and test set into 80% and 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
print(y_pred[0:5])

# Evaluate the classification tree
# Predict the test set labels
y_pred = dt.predict(X_test)

# Compute test set accuracy
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))

# ---------------------------------------------------------------------------------------------------------------------

# Logistic regression vs classification tree
# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import LogisticRegression

# Instantiate logreg
logreg = LogisticRegression(random_state=1)


def plot_labeled_decision_regions(X, y, classifier):
    # Plot decision regions
    plot_decision_regions(X, y, clf=classifier, legend=2)

    # Adding axes annotations
    plt.xlabel('"')
    plt.ylabel('Feature 1')
    plt.title('Decision boundary')

    # Show the plot
    plt.show()


# Fit logreg to the training set
logreg.fit(X_train, y_train)

# Define a list called clfs containing the two classifiers logreg and dt
clfs = [logreg, dt]

# Review the decision regions of the two classifiers
for clf in clfs:
    plot_labeled_decision_regions(X_test, y_test, clf)

# ---------------------------------------------------------------------------------------------------------------------
# Using entropy as a criterion
# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)

# Import accuracy_score
from sklearn.metrics import accuracy_score

# Use dt_entropy to predict test set labels
y_pred = dt_entropy.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)

# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)

# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', acc)

# ---------------------------------------------------------------------------------------------------------------------
# Regression tree
# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor

# Import the auto-mpg dataset
data = pd.read_csv('../../../data/supervised_learning/trees-auto-mpg.csv')

# Convert categorical data to numerical data
data = pd.get_dummies(data, drop_first=True)

# Create feature and target arrays
X = data.drop('mpg', axis=1).values
y = data['mpg'].values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Evaluate the regression tree
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = MSE(y_test, y_pred)

# Compute rmse_dt
rmse_dt = mse_dt ** 0.5

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))

# Linear regression vs regression tree
# Import LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression

# Create the regressor: lr
lr = LinearRegression()

# Fit lr to the training set
lr.fit(X_train, y_train)

# Predict test set labels
y_pred_lr = lr.predict(X_test)

# Compute mse_lr
mse_lr = MSE(y_test, y_pred_lr)

# Compute rmse_lr
rmse_lr = mse_lr ** 0.5

# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))
# ---------------------------------------------------------------------------------------------------------------------
# Bias-variance trade-off
# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Set SEED for reproducibility
SEED = 1

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)

# Evaluate the 10-fold CV error
# Compute the array containing the 10-folds CV MSEs
from sklearn.model_selection import cross_val_score

MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

# Since cross_val_score has only the option of evaluating the negative MSEs, its output should be multiplied by
# negative one to obtain the MSEs. The CV RMSE can then be obtained by computing the square root of the average MSE.

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean()) ** 0.5

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))

# Evaluate the training error
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train)) ** 0.5

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))

# ---------------------------------------------------------------------------------------------------------------------
# Define the ensemble
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNN

# import the indian liver patient dataset
data = pd.read_csv('../../../data/supervised_learning/trees_indian_liver_patient.csv')

# Convert categorical data to numerical data
data = pd.get_dummies(data, drop_first=True)

# Create feature and target arrays
X = data.drop('Dataset', axis=1).values
y = data['Dataset'].values

# Replace NaN values with the mean of the column
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Set seed for reproducibility
SEED = 1

# Instantiate lr
lr = LogisticRegression(random_state=SEED, max_iter=10000)

# Instantiate knn
knn = KNN(n_neighbors=27)

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

# Evaluate individual classifiers
# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:
    # Fit clf to the training set
    clf.fit(X_train, y_train)

    # Predict y_pred
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))

# ---------------------------------------------------------------------------------------------------------------------
# Better performance with a Voting Classifier
# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)

# Fit vc to the training set
vc.fit(X_train, y_train)

# Import accuracy_score
from sklearn.metrics import accuracy_score

# Evaluate the test set predictions
y_pred = vc.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))

# ---------------------------------------------------------------------------------------------------------------------
# Define the bagging classifier
# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = BaggingClassifier(estimator=dt, n_estimators=50, random_state=1)

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate the test set accuracy
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of bc: {:.2f}'.format(acc_test))

# ---------------------------------------------------------------------------------------------------------------------
# Prepare the ground

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)

# Instantiate bc
bc = BaggingClassifier(estimator=dt, n_estimators=50, random_state=1, oob_score=True)

# ---------------------------------------------------------------------------------------------------------------------
# OOB Score vs Test Set Score

# import the indian liver patient dataset
data = pd.read_csv('../../../data/supervised_learning/trees_indian_liver_patient.csv')

# Convert categorical data to numerical data
data = pd.get_dummies(data, drop_first=True)

# Create feature and target arrays
X = data.drop('Dataset', axis=1).values
y = data['Dataset'].values

# Replace NaN values with the mean of the column
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_test, y_pred)

# Extract the OOB accuracy from bc
acc_oob = bc.oob_score_

# Print test set accuracy
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))

# ---------------------------------------------------------------------------------------------------------------------
# Random Forests (RF)
# Train an RF regressor
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the bike dataset
data = pd.read_csv('../../../data/supervised_learning/trees_bikes.csv')

# Create feature and target arrays
X = data.drop('cnt', axis=1).values
y = data['cnt'].values

# Replace NaN values with the mean of the column

X = imputer.fit_transform(X)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25, random_state=2)

# Fit rf to the training set
rf.fit(X_train, y_train)

# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** 0.5

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_, index=data.drop('cnt', axis=1).columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.ylabel('Features')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# AdaBoost
# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# load the Indian Liver Patient dataset
data = pd.read_csv('../../../data/supervised_learning/trees_indian_liver_patient.csv')

# Convert categorical data to numerical data
data = pd.get_dummies(data, drop_first=True)

# Create feature and target arrays
X = data.drop('Dataset', axis=1).values
y = data['Dataset'].values

# Replace NaN values with the mean of the column

X = imputer.fit_transform(X)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(estimator=dt, n_estimators=180, random_state=1)

# Fit ada to the training set
ada.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:, 1]

# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))

# ---------------------------------------------------------------------------------------------------------------------
# Gradient Boosting (GB)
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Load the bike dataset
data = pd.read_csv('../../../data/supervised_learning/trees_bikes.csv')

# Create feature and target arrays
X = data.drop('cnt', axis=1).values
y = data['cnt'].values

# Replace NaN values with the mean of the column
X = imputer.fit_transform(X)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Instantiate gb
gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=2)

# Fit gb to the training set
gb.fit(X_train, y_train)

# Predict test set labels
y_pred = gb.predict(X_test)

# Compute MSE
mse_test = MSE(y_test, y_pred)

# Compute RMSE
rmse_test = mse_test ** 0.5

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))

# ---------------------------------------------------------------------------------------------------------------------
# Stochastic Gradient Boosting (SGB)
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Load the bike dataset
data = pd.read_csv('../../../data/supervised_learning/trees_bikes.csv')

# Create feature and target arrays
X = data.drop('cnt', axis=1).values
y = data['cnt'].values

# Replace NaN values with the mean of the column
X = imputer.fit_transform(X)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4, subsample=0.9, max_features=0.75, n_estimators=200, random_state=2)

# Fit sgbr to the training set
sgbr.fit(X_train, y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)

# Compute test set MSE
mse_test = MSE(y_test, y_pred)

# Compute test set RMSE
rmse_test = mse_test ** 0.5

# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))

# ---------------------------------------------------------------------------------------------------------------------
# Tuning a CART's hyperparameters
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Load the Indian Liver Patient dataset
data = pd.read_csv('../../../data/supervised_learning/trees_indian_liver_patient.csv')

# Convert categorical data to numerical data
data = pd.get_dummies(data, drop_first=True)

# Create feature and target arrays
X = data.drop('Dataset', axis=1).values
y = data['Dataset'].values

# Replace NaN values with the mean of the column

X = imputer.fit_transform(X)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Define params_dt
params_dt = {'max_depth': [2, 3, 4], 'min_samples_leaf': [0.12, 0.14, 0.16, 0.18]}

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='roc_auc', cv=5, n_jobs=-1)

# Evaluate the optimal tree
# Fit 'grid_dt' to the training set
grid_dt.fit(X_train, y_train)

# Extract the best hyperparameters
best_hyperparams = grid_dt.best_params_
print('Best hyperparameters:\n', best_hyperparams)

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))

# ---------------------------------------------------------------------------------------------------------------------
# Tuning a RF's hyperparameters
# Load the bike dataset
data = pd.read_csv('../../../data/supervised_learning/trees_bikes.csv')

# Create feature and target arrays
X = data.drop('cnt', axis=1).values
y = data['cnt'].values

# Replace NaN values with the mean of the column
X = imputer.fit_transform(X)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Instantiate rf
rf = RandomForestRegressor(random_state=2)

# Define the dictionary 'params_rf'
params_rf = {'n_estimators': [100, 350, 500], 'max_features': ['log2',  'sqrt'], 'min_samples_leaf': [2, 10, 30]}

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)

# Extract the best hyperparameters
grid_rf.fit(X_train, y_train)
best_hyperparams = grid_rf.best_params_
print('Best hyperparameters:\n', best_hyperparams)

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict the test set labels
y_pred = best_model.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** 0.5

# Print the test set RMSE
print('Test RMSE of best model: {:.3f}'.format(rmse_test))
