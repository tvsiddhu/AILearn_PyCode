import re

import pandas as pd
import numpy as np

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
    num = re.search('\d+', time_string)
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
ufo['seconds_log'] = np.log(ufo['seconds']+1)

# Print out the variance of just the seconds_log column
print("Variance of seconds_log column:", ufo['seconds_log'].var())
print("--------------------")

# Encoding categorical variables

# In this exercise, we'll take a look at the state column and transform it into dummy variables for modeling.
# We'll do this using pandas' get_dummies() function.

# Use Pandas to encode US values as 1 and others as 0
ufo['country_enc'] = ufo['country'].apply(lambda x: 1 if x == 'us' else 0)

# Print the number of unique type values
print("Number of unique type values:", len(ufo['type'].unique()))
print("--------------------")

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo['type'])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)

