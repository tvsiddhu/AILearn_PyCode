import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

import preprocessing_intro as pi

ufo = pd.read_csv('../../../data/preprocessing_data_sources/ufo_sightings_large.csv')

# Print the DataFrame info
print("Info:\n", ufo.info())
print(ufo.info())
print("--------------------")

# Change the type of the 'seconds' column to float
ufo['seconds'] = ufo['seconds'].astype(float)

# Change the date column to type datetime
ufo['date'] = pd.to_datetime(ufo['date'])

# Check the column types
print("Column types:\n", ufo.dtypes)

# Count the missing values in the length_of_time, state, and type columns, in that order
print("Missing values in length_of_time column:", ufo['length_of_time'].isnull().sum())
print("Missing values in state column:", ufo['state'].isnull().sum())
print("Missing values in type column:", ufo['type'].isnull().sum())
print("--------------------")

# Drop rows where length_of_time, state, or type are missing
ufo_no_missing = ufo.dropna(subset=['length_of_time', 'state', 'type'])

# Print out the shape of the new dataset
print("New dataset shape:", ufo_no_missing.shape)
print("--------------------")


# Extract minutes from the length_of_time column
def return_minutes(time_string):
    """
    Extracts the number of minutes from a time string.

    Parameters:
    time_string (str): The time string from which to extract minutes.

    Returns:
    int: The number of minutes extracted from the time string, or None if no digits are found.
    """
    # Convert time_string to string
    time_string = str(time_string)
    # Search for a digit in the time_string
    num = re.search(r'\d+', time_string)
    if num is not None:
        return int(num.group())


# Apply the extraction to the length_of_time column
ufo['minutes'] = ufo['length_of_time'].apply(lambda row: return_minutes(row))

# Take a look at the head of both of the columns
print("Head of length_of_time column:", ufo['length_of_time'].head())
print("Head of minutes column:", ufo['minutes'].head())
print("--------------------")

# Identifying features for standardization

# In this exercise, you'll investigate the variance of columns in the UFO dataset to determine which features should
# be standardized. After taking a look at the variances of the seconds and minutes column, you'll see that the
# variance of the seconds column is extremely high. Because seconds and minutes are related to each other (an issue
# we'll deal with when we select features for modeling), let's log normalize the seconds' column.

# Check the variance of the seconds and minutes columns
print("Variances:\n", ufo[['seconds', 'minutes']].var())
print("--------------------")

# Log normalize the seconds column
ufo['seconds_log'] = np.log(ufo['seconds'] + 1)

# Print out the variance of just the seconds_log column
print("Variance of seconds_log column:", ufo['seconds_log'].var())
print("--------------------")

# Encoding categorical variables

# In this exercise, we'll take a look at the state column and transform it into dummy variables for modeling.
# We'll do this using pandas' get_dummies() function.

# Use Pandas to encode US values as 1 and others as 0 (Binary Encoding)
ufo['country_enc'] = ufo['country'].apply(lambda x: 1 if x == 'us' else 0)

# Print the number of unique type values
print("Number of unique type values:", len(ufo['type'].unique()))
print("--------------------")

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo['type'])

# Concatenate this set back to the ufo DataFrame (One-hot Encoding)
ufo = pd.concat([ufo, type_set], axis=1)

# Features from dates

# Another feature engineering task to perform is month and year extraction. Perform this task on the date column of
# the ufo dataset.

# Look at the first 5 rows of the date column
print("First 5 rows of date column:\n", ufo['date'].head())
print("--------------------")

# Extract the month from the date column
ufo['month'] = ufo['date'].apply(lambda row: row.month)

# Extract the year from the date column
ufo['year'] = ufo['date'].apply(lambda row: row.year)

# Take a look at the head of all three columns
print("Head of date column:", ufo['date'].head())
print("Head of month column:", ufo['month'].head())
print("Head of year column:", ufo['year'].head())
print("--------------------")

# Text vectorization

# Let's transform the desc column in the UFO dataset into tf/idf vectors, since there's likely something we can learn
# from this field.


# Take a look at the head of the desc field
print("Head of desc column:", ufo['desc'].head())
print("--------------------")

# Create a TfidfVectorizer object
vec = TfidfVectorizer()

# Fill NaN values in the desc column
ufo['desc'] = ufo['desc'].fillna('')

# Use vec's fit_transform method on the desc field
desc_tfidf = vec.fit_transform(ufo['desc'])

# Look at the number of columns this creates
print("Number of columns created:", desc_tfidf.shape[1])
print("--------------------")

# Selecting the ideal dataset

# Now to get rid of some of the unnecessary features in the ufo dataset. Because the country column has been encoded
# as country_enc, you can select it and drop the other columns related to location: city, country, lat, long, and state.
# You've engineered the month and year columns, so you no longer need the date or recorded columns. You also
# standardized the seconds column as seconds_log, so you can drop seconds and minutes.
# You vectorized desc, so it can be removed. For now, you'll keep type.
# You can also get rid of the length_of_time column, which is unnecessary after extracting minutes

# Make a list of features to drop
to_drop = ['city', 'country', 'lat', 'long', 'state', 'date', 'recorded', 'desc', 'length_of_time', 'seconds',
           'minutes']

# Drop those features
ufo_dropped = ufo.drop(to_drop, axis=1).fillna(0)

# Let's also filter some words out of the text vector we created
vocab = {v: k for k, v in vec.vocabulary_.items()}
filtered_words = pi.words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)

# Modeling the UFO dataset, part 1

# In this exercise, you're going to build a k-nearest neighbor model to predict which country the UFO sighting took
# place in. The X dataset contains the log-normalized seconds column, the one-hot encoded type columns, as well as
# the month and year when the sighting took place. The y labels are the encoded country column, where 1 is "us" and 0
# is "ca".

# X = ufo.drop(['country_enc', 'type', 'seconds', 'desc', 'date'], axis=1)
# Change the type of the 'seconds' column to float

# to check - X = ufo_dropped.select_dtypes(include=[np.number])

X = ufo_dropped.drop(['country_enc', 'type'], axis=1)
y = ufo_dropped['country_enc']

# Take a look at the features in the X set of data
print("Features in X set:\n", X.columns)
print("--------------------")

print("Data types in X set:\n", X.dtypes)
print("--------------------")

# Split the X and y sets using train_test_split, setting stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Fit knn to the training sets
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Model performance
print("KNN Model score:", knn.score(X_test, y_test))
print("--------------------")

# Modeling the UFO dataset, part 2

# Finally, let's build a model using the text vector we created, desc_tfidf, using the filtered_words list to create a
# filtered text vector. Let's see if we can predict the type of the sighting based on the text. We'll use a Naive Bayes
# model for this.

# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y
X_train, X_test, y_train, y_test = train_test_split(filtered_text.toarray(), y, stratify=y, random_state=42)

# Fit nb to the training sets
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Print the score of nb on the test sets
print("Naive Bayes Model score:", nb.score(X_test, y_test))
print("--------------------")
