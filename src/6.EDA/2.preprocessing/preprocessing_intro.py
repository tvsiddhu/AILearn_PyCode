# 1. Dropping missing data
# Now that you've explored the volunteer dataset and understand its structure and contents,
# it's time to begin dropping missing values.
# In this exercise, you'll drop both columns and rows to create a subset of the volunteer dataset.

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

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


# Create the StandardScaler object
scaler = StandardScaler()

# Subset the wine dataset to only the 'Ash', 'Alcalinity of ash', and 'Magnesium' columns
wine_subset = wine[['Ash', 'Alcalinity of ash', 'Magnesium']]

# Apply the scaler to the wine subset
wine_subset_scaled = scaler.fit_transform(wine_subset)

# 9. KNN on non-scaled data
# Let's first take a look at the accuracy of a K-nearest neighbors model on the wine dataset without
# standardizing the data. The knn model as well as the X_train and y_train datasets have been created already.
# The X_train dataset has been scaled, but the X_test data has not been scaled.


# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(wine[['Ash', 'Alcalinity of ash', 'Magnesium']],
                                                    wine['Type'], random_state=42)
# Fit the k-nearest neighbors model to the training data
knn.fit(X_train, y_train)

# Score the k-nearest neighbors model on the test data
print(knn.score(X_test, y_test))

# 10. KNN on scaled data
# The accuracy score on the unscaled wine dataset was decent, but we can likely do better if we scale the dataset.
# The process is mostly the same as the previous exercise, with the added step of scaling the X data. Once again,
# the knn model as well as the X_train and y_train datasets have already been created for you.

# Create the StandardScaler object
scaler = StandardScaler()

# Scale the training and test features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train_scaled, y_train)

# Score the k-nearest neighbors model on the test data
print("Score the k-nearest neighbors model on the test data")
print(knn.score(X_test_scaled, y_test))

# 11. Encoding categorical variables - binary
# Take a look at the hiking dataset. There are several columns here that
# need encoding before they can be modeled, one of which is the Accessible column. Accessible is a binary feature,
# so it has two values, Y or N, which need to be encoded into 1's and 0's. Use the scikit-learn LabelEncoder method to
# perform this transformation.

hiking = pd.read_csv('../../../data/preprocessing_data_sources/hiking.csv')

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking['Accessible'])

# Compare the two columns
print(hiking[['Accessible', 'Accessible_enc']].head())

# 12. Encoding categorical variables - one-hot
# One of the columns in the volunteer dataset, category_desc, gives category descriptions for the volunteer
# opportunities listed. Because it is a categorical variable with more than two categories, we need to use one-hot
# encoding to transform this column numerically. Use pandas' pd.get_dummies() function to do so.

# Transform the category_desc column
category_enc = pd.get_dummies(volunteer['category_desc'])

# Take a look at the encoded columns
print(category_enc.head())

# 13. Aggregating numerical features
# A good use case for taking an aggregate statistic to create a new feature is when you have many features with
# similar, related values. Here, you have a DataFrame of running times named running_times_5k. For each name in the
# dataset, take the mean of their 5 run times.

running_times_5k = pd.read_csv('../../../data/preprocessing_data_sources/running_times_5k.csv')

# Use .loc to create a new column, mean of the 5 columns
running_times_5k['mean'] = running_times_5k.loc[:, 'run1':'run5'].mean(axis=1)

# Take a look at the results
print(running_times_5k.head())

# 14. Extracting datetime components

# There are several columns in the volunteer dataset comprised of date-times. Let's take a look at the start_date_date
# column and extract just the month to use as a feature for modeling.

# First, we'll need to convert the column to a datetime object.
# Then we'll use the .dt accessor to grab just the month.

# Convert start_date_date to a datetime object
volunteer['start_date_converted'] = pd.to_datetime(volunteer['start_date_date'])

# Extract just the month from the converted column
volunteer['start_date_month'] = volunteer['start_date_converted'].dt.month

# Take a look at the original and new columns
print(volunteer[['start_date_converted', 'start_date_month']].head())


# 15. Extracting string patterns

# The Length column in the hiking dataset is a column of strings, but contained in the column is the mileage for the
# hike. We're going to extract this mileage using regular expressions, and then use a lambda in pandas to apply the
# extraction to the DataFrame.

# Write a pattern that will extract numbers and decimals from text

def return_mileage(length):
    """
    Extracts mileage from a string using a regular expression.

    Parameters:
    length (str): The string containing the mileage information.

    Returns:
    float: The extracted mileage as a float if a match is found, otherwise None.
    """
    # Compile a regular expression pattern to match numbers with decimals
    pattern = re.compile(r'\d+\.\d+')

    # Search the text for matches using the compiled pattern
    mile = re.match(pattern, length)

    # If a match is found, return the matched value as a float
    if mile is not None:
        return float(mile.group(0))


# Apply the function to the Length column and take a look at both columns
hiking['Length_num'] = hiking['Length'].apply(lambda row: return_mileage(row))
print("-------------------")
print("Extracting string patterns")
print(hiking[['Length', 'Length_num']].head())

# 16. Vectorizing text

# You'll now transform the volunteer dataset's title column into a text vector, which you'll use in a prediction task
# in the next exercise.

# Take the title text
title_text = volunteer['title']

# Create the vectorizer method
title_text = title_text.fillna('')  # fill NaN values with an empty string
title_text = title_text.str.lower()  # convert to lowercase
title_text = title_text.str.replace(r'[^\w\s]', '')  # remove punctuation

tfidf_vec = TfidfVectorizer()

text_tfidf = tfidf_vec.fit_transform(title_text)
print("-------------------")
print("Vectorizing text")
print(text_tfidf.shape)

# 17. Text classification using tf-idf

# Text classification using tf/idf vectors

# Now that you've encoded the volunteer dataset's title column into tf/idf vectors, you'll use those vectors to
# predict the category_desc column.

# Split the text_tfidf vector and y target variable into training and test sets, setting the stratify parameter equal
# to y, since the class distribution is uneven. Notice that we have to run the .toarray() method on the tf/idf
# vector, in order to get in it the proper format for scikit-learn. Fit the X_train and y_train data to the Naive
# Bayes model, nb.
# Print out the test set accuracy.

# Instantiate the Naive Bayes classifier
nb = MultinomialNB()

# Split the dataset according to the class distribution of category_desc
y = volunteer["category_desc"].fillna('')
X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), y, stratify=y, random_state=42)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print("-------------------")
print("Text classification using tf-idf vectors")
print(nb.score(X_test, y_test))

# 18. Selecting relevant features

# You'll reduce the feature set by removing some of the columns that are not providing useful information for the
# prediction task. The goal is to run a machine learning model with a smaller feature set and compare the accuracy to
# the model created with all the features.

# In this exercise, you'll identify the redundant columns in the volunteer dataset, and perform feature selection on
# the dataset to return a DataFrame of the relevant features. For example, if you explore the volunteer dataset in
# the console, you'll see three features which are related to location: locality, region, and postalcode. They
# contain related information, so it would make sense to keep only one of the features.

volunteer = pd.read_csv('../../../data/preprocessing_data_sources/volunteer_feature_selection.csv')

# Create a list of redundant column names to drop
to_drop = ['category_desc', 'created_date', 'locality', 'region', 'vol_requests']

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis=1)

# Print out the head of the new dataset
print("-------------------")
print("Selecting relevant features")
print(volunteer_subset.head())

# 19. Checking for correlated features

# You'll now return to the wine dataset, which consists of continuous, numerical features. Run Pearson's correlation
# coefficient on the dataset to determine which columns are good candidates for eliminating. Then, remove those
# columns from the DataFrame.

wine = pd.read_csv('../../../data/preprocessing_data_sources/wine_types.csv')

# Print out the column correlations of the wine dataset
print(wine.corr())

# Drop that column from the DataFrame
wine = wine.drop('Flavanoids', axis=1)

print("-------------------")
print("Checking for correlated features")
print(wine.head())

# 20. Exploring text vectors, part 1

# Let's expand on the text vector exploration method we just learned about, using the volunteer dataset's title
# tf/idf vectors. In this first part of text vector exploration, we'll add to that function in order to return the
# average feature value for any token.

# Add parameters called original_vocab, for the tfidf_vec.vocabulary_, and top_n. Call pd.Series() on the zipped
# dictionary. This will make it easier to operate on. Use the .sort_values() function to sort the series and slice
# the index up to top_n words. Call the function, setting original_vocab=tfidf_vec.vocabulary_,
# setting vector_index=8 to grab the 9th row, and setting top_n=3, to grab the top 3 weighted words.

volunteer = pd.read_csv('../../../data/preprocessing_data_sources/volunteer_text_vector_1.csv')

vocab = {v: k for k, v in tfidf_vec.vocabulary_.items()}


# Add in the rest of the arguments
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    """
    Returns the top_n weighted words from a tf-idf vector.

    Parameters:
    vocab (dict): The vocabulary mapping of the tf-idf vectorizer.
    original_vocab (dict): The original vocabulary of the tf-idf vectorizer.
    vector (scipy.sparse.csr.csr_matrix): The tf-idf vector.
    vector_index (int): The index of the vector to analyze.
    top_n (int): The number of top weighted words to return.

    Returns:
    list: A list of the top_n weighted words.
    """
    # Create a dictionary of the indices and data of the specified vector
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))

    # Transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]: zipped[i] for i in vector[vector_index].indices})

    # Sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]


# Print out the weighted words
print("-------------------")
print("Exploring text vectors, part 1")
print(return_weights(vocab=vocab, original_vocab=tfidf_vec.vocabulary_, vector=text_tfidf, vector_index=8, top_n=3))


# 21. Exploring text vectors, part 2

# Using the return_weights() function you wrote in the previous exercise, you're now going to extract the top words
# from each document in the text vector, return a list of the word indices, and use that list to filter the text
# vector down to those top words.

def words_to_filter(vocab, original_vocab, vector, top_n):
    """
    Extracts the top_n words from each document in the tf-idf vector and returns a set of unique word indices.

    Parameters:
    vocab (dict): The vocabulary mapping of the tf-idf vectorizer.
    original_vocab (dict): The original vocabulary of the tf-idf vectorizer.
    vector (scipy.sparse.csr.csr_matrix): The tf-idf vector.
    top_n (int): The number of top weighted words to extract from each document.

    Returns:
    set: A set of unique word indices.
    """
    filter_list = []
    for i in range(0, vector.shape[0]):
        # Call the return_weights function and extend filter_list
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)

    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)


# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab=vocab, original_vocab=tfidf_vec.vocabulary_, vector=text_tfidf, top_n=3)

# Filter the columns in text_tfidf to only include the filtered words
filtered_text = text_tfidf[:, list(filtered_words)]

# Print out the results
print("-------------------")
print("Exploring text vectors, part 2")
print(filtered_text.shape)

# 22. Training Naive Bayes with feature selection

# You'll reduce the feature set by removing some of the columns that are not providing useful information for the
# prediction task. You'll now re-run the Naive Bayes text classification model that you ran at the end of Chapter 3
# with our selection choices from the previous exercise: the volunteer dataset's title and category_desc columns.

# Split the dataset according to the class distribution of category_desc, setting stratify=y.
X_train, X_test, y_train, y_test = train_test_split(filtered_text.toarray(), y, stratify=y, random_state=42)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print("-------------------")
print("Training Naive Bayes with feature selection")
print(nb.score(X_test, y_test))

# 23. Using PCA

# Principal component analysis (PCA) is a statistical method that summarizes large data sets into a smaller set of
# variables, called principal components. PCA is a popular multivariate statistical technique that is used in many
# scientific disciplines to analyze data. PCA is used to decompose a multivariate dataset in a set of successive
# orthogonal components that explain a maximum amount of the variance. In scikit-learn, PCA is implemented as a
# transformer object that learns n components in its fit method, and can be used on new data to project it on these
# components.

# Let's apply PCA to the wine dataset, to see if we can get an increase in our model's accuracy.

# Instantiate the PCA object and fit to the wine data
pca = PCA()

# Define the features and labels from the wine dataset
X = wine.drop('Type', axis=1)
y = wine['Type']

# Split the transformed X and y data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y ,random_state=42)

# Apply PCA to wine dataset X data
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)

# Look at the percentage of variance explained by the different components
print("-------------------")
print("Using PCA")
print(pca.explained_variance_ratio_)
print(pca_X_train.shape)

# 24. Training a model with PCA

# Now that you have run PCA on the wine dataset, you'll re-run the k-NN model from Chapter 2 to see if it
# improves the model's accuracy.

# Instantiate the k-NN model
knn = KNeighborsClassifier()

# Fit the k-NN model to the training data
knn.fit(pca_X_train, y_train)

# Score the k-NN model on the test data
print("-------------------")
print("Training a model with PCA")
print(knn.score(pca_X_test, y_test))

