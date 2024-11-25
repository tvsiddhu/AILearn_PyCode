# Machine learning with scikit-learn
# Supervised learning - Regression & Linear Regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load the sales dataset
sales_df = pd.read_csv('../../data/supervised_learning/advertising_and_sales.csv')

# Create X from the radio column's values
X = sales_df["radio"].values
# Create y from the sales column's values
y = sales_df["sales"].values

# Reshape X and y
X = X.reshape(-1, 1)
# y = y.reshape(-1, 1)

# Check the shape of X and y
print(X.shape)
print(y.shape)

# Create a linear regression model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Check the model's predictions
predictions = reg.predict(X)

print("Predictions: ", predictions[:5])

# Plot the data
plt.scatter(X, y, color='blue')
plt.plot(X, predictions, color='red', linewidth=3)
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Show the plot
plt.show()

# Fit and predict for regression

# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df.sales.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
"----------------------------------------------------------------------------------------------------------------------"
# Instantiate a Linear regression model
reg_all = LinearRegression()

# Fit the model to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data
y_pred = reg_all.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
# Regression Performance
# Compute and print the R^2 and RMSE
r_squared = reg_all.score(X_test, y_test)

# Compute the mean squared error of the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the R^2 and RMSE
print("R^2: {}".format(r_squared))
print("Root Mean Squared Error: {}".format(rmse))
"----------------------------------------------------------------------------------------------------------------------"
# Cross-validation
# Create a K-Fold cross-validation object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print the cross-validation scores
print("Cross-validation scores: {}".format(cv_scores))
"----------------------------------------------------------------------------------------------------------------------"
# Analyzing the cross-validation results

# Print the mean
print("Average 6-Fold CV Score: {}".format(np.mean(cv_scores)))

# Print the standard deviation
print("Standard deviation of 6-Fold CV scores: {}".format(np.std(cv_scores)))

# Print the 95% confidence interval
print("95% confidence interval: {}".format(np.percentile(cv_scores, [2.5, 97.5])))
"----------------------------------------------------------------------------------------------------------------------"
# Regularized regression (Ridge regression)

alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []

for alpha in alphas:
    # Instantiate a Ridge regressor
    ridge = Ridge(alpha=alpha)

    # Fit the model to the data
    ridge.fit(X_train, y_train)

    # Obtain R^2 score
    score = ridge.score(X_test, y_test)
    ridge_scores.append(score)
print("Ridge Scores: ", ridge_scores)
"----------------------------------------------------------------------------------------------------------------------"
# Lasso regression for feature selection
# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print("Lasso Coefficients: ", lasso_coef)
plt.bar(range(len(sales_df.columns) - 1), lasso_coef)
plt.xticks(range(len(sales_df.columns) - 1), sales_df.columns[:-1], rotation=45)
plt.ylabel("Coefficients")
plt.show()
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
# Assessing a diabetes prediction classifier

# Load the diabetes dataset
diabetes_df = pd.read_csv('../../data/supervised_learning/diabetes_clean.csv')

knn = KNeighborsClassifier(n_neighbors=6)

# Create arrays for the features and the response variable
y = diabetes_df['diabetes'].values
X = diabetes_df.drop('diabetes', axis=1).values

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("Classification Report: ")
print(classification_report(y_test, y_pred))
"----------------------------------------------------------------------------------------------------------------------"
# Building a logistic regression model
# Instantiate a Logistic Regression model

logreg = LogisticRegression()

# Fit the model to the data
logreg.fit(X_train, y_train)

# Predict the probabilities of each individual in the test set having a diabetes diagnosis, storing the array of
# positive probabilities as y_pred_probs.

y_pred_probs = logreg.predict_proba(X_test)[:, 1]

print("Predicted Probabilities: ", y_pred_probs[:10])

# The ROC curve

# Generate ROC Curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()
"----------------------------------------------------------------------------------------------------------------------"

# AUC computation
# Import the roc_auc_score function

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_probs)))

# Calculate the confusion matrix and classification report
y_pred = logreg.predict(X_test)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("Classification Report: ")
print(classification_report(y_test, y_pred))
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
# Hyperparameter tuning with GridSearchCV

# Set up the hyperparameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)

# Fit to the training set
lasso_cv.fit(X_train, y_train)

# Predict the labels of the test set
print("Tuned lasso parameters: {}".format(lasso_cv.best_params_))
print(" Tuned lasso score: {}".format(lasso_cv.best_score_))
"----------------------------------------------------------------------------------------------------------------------"
# Hyperparameter tuning with RandomizedSearchCV

# Import RandomizedSearchCV

# Create the parameter grid
# Check the parameter space
# params = {
#     "penalty": ["l1", "l2"],
#     "tol": np.linspace(0.0001, 1.0, 50),  # Make sure to include 'tol' in the params
#     "C": np.linspace(0.1, 1.0, 50),
#     "class_weight": ["balanced", {0:0.8, 1:0.2}]
# }

params = {
    "penalty": ["l2"],
    "tol": np.linspace(0.0001, 1.0, 50),
    "C": np.linspace(0.1, 1.0, 50),
    "class_weight": ["balanced", {0: 0.8, 1: 0.2}]
}

# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)

# Fit it to the data
logreg_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))

# Processing data
# load the music dataset
music_df = pd.read_csv('../../data/supervised_learning/music_clean.csv')
music_dummies = pd.get_dummies(music_df, drop_first=True)

# Print the new dataframe's shape
print("Shape of music_dummies: {}".format(music_dummies.shape))
"----------------------------------------------------------------------------------------------------------------------"
# Regression with categorical features Create X and y arrays -  X, containing all features in music_dummies, and y,
# consisting of the "popularity" column, respectively.
X = music_dummies.drop("popularity", axis=1).values
y = music_dummies.popularity.values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

kf = KFold(n_splits=6, shuffle=True, random_state=42)

# Instantiate a Ridge regressor
ridge = Ridge(alpha=0.2)

# Fit the model to the training data
ridge.fit(X_train, y_train)

# Predict on the test data
y_pred = ridge.predict(X_test)

# Perform cross-validation
scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")

# Print the cross-validated scores
print("Cross-validated scores: {}".format(scores))

# Calculate RMSE
rmse = np.sqrt(-scores)
print("Root Mean Squared Error: {}".format(rmse))
print("Average RMSE: {}".format(np.mean(rmse)))
print("Standard Deviation of the target array: {}".format(np.std(y)))
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"

# Handling missing data
# Print the number of missing values in music_df
music_df = pd.read_csv('../../data/supervised_learning/music_raw.csv')
# Check for missing values in the DataFrame
missing_values = music_df.isna()

# Print missing values for each column
print("Missing values in music_df: ", missing_values.sum().sort_values(ascending=True))
print("missing value %: ", (missing_values.sum().sort_values(ascending=True) / music_df.shape[0]) * 100)

# Remove values where less than 5% of the data is missing
music_df = music_df.dropna(subset=["genre", "popularity", "loudness", "liveness", "tempo"])

# Convert genre to a binary column
music_df["genre"] = music_df["genre"].apply(lambda x: 1 if x == "Rock" else 0)

# Print the shape of the new DataFrame
print(music_df.isna().sum().sort_values())
print("Shape of the `music_df`: {}".format(music_df.shape))
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"

# Pipeline for song genre prediction
# Import SimpleImputer and Pipeline
# Instantiate SimpleImputer
imputer = SimpleImputer(strategy="mean")

# Instantiate a knn model
knn = KNeighborsClassifier(n_neighbors=3)

# Create steps, a list of tuples containing the
# imputer variable you created, called "imputer", followed by the knn model you created, called "knn".
steps = [("imputer", imputer), ("knn", knn)]

# Create a pipeline using the steps
pipeline = Pipeline(steps)

# Split the data into training and test sets
X = music_df.drop("genre", axis=1).values
y = music_df.genre.values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute the confusion matrix and classification report
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("Classification Report: ")
print(classification_report(y_test, y_pred))
"----------------------------------------------------------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"
# Centering and scaling your data
#
# Given any column, we can subtract the mean and divide by the variance so that all
# features are centered around zero and have a variance of one. This is called standardization. We can also subtract
# the minimum and divide by the range of the data so the normalized dataset has minimum zero and maximum one. Or,
# we can center our data so that it ranges from -1 to 1 instead.
"----------------------------------------------------------------------------------------------------------------------"
# Centering and scaling for regression
# Instantiate a StandardScaler object
scaler = StandardScaler()

music_df = pd.read_csv('../../data/supervised_learning/music_clean.csv')

X = music_df.drop("loudness", axis=1).values
y = music_df.loudness.values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the scaler to the training data and transform the test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the pipeline steps
steps = [("scaler", StandardScaler()), ("lasso", Lasso(alpha=0.5))]

# Instantiate the pipeline
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute and print R^2 and RMSE
r_squared = pipeline.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("R^2: {}".format(r_squared))
print("Root Mean Squared Error: {}".format(rmse))

"----------------------------------------------------------------------------------------------------------------------"

# Centering and scaling for classification
# Build the steps for the pipeline
music_df = pd.read_csv('../../data/supervised_learning/music_clean.csv')

X = music_df.drop("genre", axis=1).values
y = music_df.genre.values

steps = [("scaler", StandardScaler()), ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)

# Create the parameter grid
parameters = {"logreg__C": np.linspace(0.001, 1, 20)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
print("Tuned Model Score: {}".format(cv.best_score_))
"----------------------------------------------------------------------------------------------------------------------"
# Evaluating multiple models

music_df = pd.read_csv('../../data/supervised_learning/music_clean.csv')

X = music_df.drop("energy", axis=1).values
y = music_df.energy.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Visualizing regression model performance
# Create a list of regressors
models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []

# Loop over the models
for model in models.values():
    # Fit the model to the data
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)

    kf = KFold(n_splits=6, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf)

    # Append the model and its score to the results list
    results.append(cv_scores)

# Print the results
print(results)

# Create a boxplot of the cross-validation scores
model_names = list(models.keys())
cv_scores = [cross_val_score(model, X, y, cv=kf) for model in models.values()]

plt.boxplot(results, tick_labels=model_names)
plt.xlabel('Model')
plt.ylabel('Cross-Validation Score')
plt.title('Boxplot of Cross-Validation Scores for Different Models')
plt.show()
"----------------------------------------------------------------------------------------------------------------------"

# Predict on the test set and compute metrics
models = {'Linear Regression': LinearRegression(), 'Ridge': Ridge()}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate the test_rmse
    test_rmse = mean_squared_error(y_test, y_pred)
    print("{} Test Set RMSE: {}".format(name, test_rmse))
"----------------------------------------------------------------------------------------------------------------------"
# Visualizing classification model performance

music_df = pd.read_csv('../../data/supervised_learning/music_clean.csv')
music_df["popularity"] = music_df["popularity"].apply(lambda x: 1 if x >= music_df["popularity"].median() else 0)

X = music_df.drop("popularity", axis=1).values
y = music_df.popularity.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train = y_train.ravel()
y_test = y_test.ravel()

# Create models dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=6),
    "Decision Tree Classifier": DecisionTreeClassifier()
}
results = []

# Loop through the models' values
for model in models.values():

    # Instantiate a KFold object with 6 splits, shuffling, and a random state of 12
    kf = KFold(n_splits=6, shuffle=True, random_state=12)

    # Perform cross-validation
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)
plt.boxplot(results, tick_labels=list(models.keys()))
plt.xlabel('Model')
plt.ylabel('Cross-Validation Score')
plt.title('Boxplot of Cross-Validation Scores for Different Models')
plt.show()
"----------------------------------------------------------------------------------------------------------------------"
# Pipeline for predicting song popularity
# create a pipeline with the steps: imputer, scaler, and logreg
steps = [("imp_mean", SimpleImputer()), ("scaler", StandardScaler()), ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)

# Create the parameter grid
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
         "logreg__C": np.linspace(0.001, 1.0, 10)}

# Instantiate a GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params, cv=5)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)

# Compute and print metrics
print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_, tuning.score(X_test, y_test)))
