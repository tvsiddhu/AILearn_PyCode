# 1. Dropping missing data
# Now that you've explored the volunteer dataset and understand its structure and contents,
# it's time to begin dropping missing values.
# In this exercise, you'll drop both columns and rows to create a subset of the volunteer dataset.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
volunteer = pd.read_csv('../../../data/preprocessing_data_sources/volunteer_opportunities.csv')
print(volunteer.columns)
print(volunteer.shape)
print(volunteer.head())

# Drop the latitude and longitude columns from the dataset
volunteer_cols = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Drop rows with missing category_desc values from volunteer_cols_subset
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])

# Print out the shape of the subset
print(volunteer_subset.shape)

# 2. Converting a column type
# If you take a look at the volunteer dataset types, you'll see that the column hits is
# type object. But, if you actually look at the column, you'll see that it consists of integers. Let's convert that
# column to type int.

# Print the head of the hits column
print(volunteer['hits'].head())

# Convert the hits column to type int
volunteer['hits'] = volunteer['hits'].astype(int)

# Look at the dtypes of the dataset
print(volunteer.dtypes)

# 3. Class imbalance
# In the volunteer dataset, you're thinking about trying to predict the category_desc variable
# using the other features in the dataset. First, though, you need to know what the class distribution (and
# imbalance) is for that label.

# Which descriptions occur less than 50 times in the volunteer dataset?
# Print the category_desc value counts
print(volunteer['category_desc'].value_counts())

# 4. Stratified sampling
# You now know that the distribution of class labels in the category_desc column of the
# volunteer dataset is uneven. If you wanted to train a model to predict category_desc, you'll need to ensure that
# the model is trained on a sample of data that is representative of the entire dataset. Stratified sampling is a way
# to achieve this!

volunteer_stratified = pd.read_csv('../../../data/preprocessing_data_sources/volunteer_stratified_sampling.csv')

# Create a data with all columns except category_desc
X = volunteer_stratified.drop('category_desc', axis=1)

# Create a category_desc labels dataset
y = volunteer_stratified[['category_desc']]

# Use stratified sampling to split up the dataset according to the volunteer_y dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Print out the category_desc counts on the training y labels
print(y_train['category_desc'].value_counts())

# 5. Modeling without normalizing
# Let's take a look at what might happen to your model's accuracy if you try to model
# data without doing some sort of standardization first.
#
# Here we have a subset of the wine dataset. One of the columns, Proline, has an extremely high variance compared to
# the other columns. This is an example of where a technique like log normalization would come in handy, which you'll
# learn about in the next section.
#
# The scikit-learn model training process should be familiar to you at this point, so we won't go too in-depth with
# it. You already have a k-nearest neighbors model available (knn) as well as the X and y sets you need to fit and
# score on.

knn = KNeighborsClassifier()


# Encode the category_desc column
le = LabelEncoder()
volunteer_stratified['category_desc_encoded'] = le.fit_transform(volunteer_stratified['category_desc'])

# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    volunteer_stratified[['category_desc_encoded']], volunteer_stratified[['category_desc_encoded']], random_state=42)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train, y_train)

# Score the k-nearest neighbors model on the test data
print(knn.score(X_test, y_test))

# 6. Log normalization
wine = pd.read_csv('../../../data/preprocessing_data_sources/wine_types.csv')

# Print out the variance of the Proline column
print("Print out the variance of the Proline column :")
print(wine['Proline'].var())

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine['Proline'])

# Check the variance of the Proline column again
print("Check the variance of the Proline column again:")
print(wine['Proline_log'].var())

# 7. Scaling data - Investigating columns
# We want to use the Ash, Alcalinity of ash, and Magnesium columns in the wine dataset to train a linear model on.
# These columns are on different scales, so we need to normalize them first. Here, we will compare the effect of
# normalizing using different methods by looking at the variance of the columns.

# Print out the variance of the Ash, Alcalinity of ash, and Magnesium columns
print("Print out the variance of the Ash, Alcalinity of ash, and Magnesium columns:")
print(wine[['Ash', 'Alcalinity of ash', 'Magnesium']].var())

print("Print out the standard deviation of the Ash, Alcalinity of ash, and Magnesium columns:")
print(wine[['Ash', 'Alcalinity of ash', 'Magnesium']].std())


# 8. Scaling data - Standardizing columns
# Since we know that the Ash, Alcalinity of ash, and Magnesium columns in the
# wine dataset are all on different scales, let's standardize them in a way that allows for use in a linear model.

# Import StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler

# Create the StandardScaler object
scaler = StandardScaler()

# Subset the wine dataset to only the 'Ash', 'Alcalinity of ash', and 'Magnesium' columns
wine_subset = wine[['Ash', 'Alcalinity of ash', 'Magnesium']]

# Apply the scaler to the wine subset
wine_subset_scaled = scaler.fit_transform(wine_subset)

