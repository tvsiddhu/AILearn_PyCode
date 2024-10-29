# Machine learning with scikit-learn
# Supervised learning - Classification & K-Nearest Neighbors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Read the data
churn_df = pd.read_csv('../../data/supervised_learning/telecom_churn_1.csv')
print(churn_df.head())

# 2. Create feature and target arrays
X = churn_df["churn"].values
y = churn_df[["account_length", "customer_service_calls"]].values

# 3. Create a K-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# 4. Fit the classifier to the data
knn.fit(y, X)

X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

# 5. Predict the labels for the training data X_new
y_pred = knn.predict(X_new)
print("Prediction: {}".format(y_pred))

# 6. Split the data
X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Create a K-NN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# 8. Fit the classifier to the training data
knn.fit(X_train, y_train)

# 9. Print the accuracy
print(knn.score(X_test, y_test))

# 10. Create neighbors
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

# 11. Compute the accuracy
train_accuracies[12] = knn.score(X_train, y_train)
test_accuracies[12] = knn.score(X_test, y_test)
print(neighbors, "\n", train_accuracies, "\n", test_accuracies)

# 12. Visualizing model complexity

# Add a title
plt.title('k-NN: Varying Number of Neighbors')

# Plot training accuracies
plt.plot(neighbors, list(train_accuracies.values()), label='Training Accuracy')

# Plot testing accuracies
plt.plot(neighbors, list(test_accuracies.values()), label='Testing Accuracy')

# Add labels
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()

# Display the plot
plt.show()
