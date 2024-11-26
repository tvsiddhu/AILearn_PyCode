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
