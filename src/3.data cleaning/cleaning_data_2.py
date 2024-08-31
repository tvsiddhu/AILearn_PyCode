import pandas as pd
import numpy as np
# Finding inconsistency In this exercise and throughout this chapter, you'll be working with the airlines DataFrame
# which contains survey responses on the San Francisco Airport from airline customers.
#
# The DataFrame contains flight metadata such as the airline, the destination, waiting times as well as answers to
# key questions regarding cleanliness, safety, and satisfaction. Another DataFrame named categories was created,
# containing all correct possible values for the survey columns.
#
# In this exercise, you will use both of these DataFrames to find survey answers with inconsistent values,
# and drop them, effectively performing an outer and inner join on both these DataFrames as seen in the video
# exercise. The pandas package has been imported as pd, and the airlines and categories DataFrames are in your
# environment.

airlines = pd.read_csv('../../data/cleaning_data_sources/airlines_new.csv')
categories = airlines[['cleanliness', 'safety', 'satisfaction']].copy()

# print Airlines DataFrame
print(airlines)

# Print categories DataFrame
print(categories)

# Print unique values of survey columns in airlines
print('Cleanliness: ', airlines['cleanliness'].unique(), "\n")
print('Safety: ', airlines['safety'].unique(), "\n")
print('Satisfaction: ', airlines['satisfaction'].unique(), "\n")

# Find the cleanliness category in airlines not in categories
print("-------------------")
cat_clean = set(airlines['cleanliness']).difference(categories['cleanliness'])

# Find rows with that category
cat_clean_rows = airlines['cleanliness'].isin(cat_clean)

# Print rows with inconsistent category
print(airlines[cat_clean_rows])

# Finding consistency
#
# In this exercise and throughout this chapter, you'll be working with the airlines DataFrame which contains survey
# responses on the San Francisco Airport from airline customers.
#
# The DataFrame contains flight metadata such as the airline, the destination, waiting times as well as answers to
# key questions regarding cleanliness, safety, and satisfaction. Another DataFrame named categories was created,
# containing all correct possible values for the survey columns.
#
# In this exercise, you will use both of these DataFrames to find survey answers with inconsistent values,
# and drop them, effectively performing an outer and inner join on both these DataFrames as seen in the video exercise.

# Find the cleanliness category in airlines not in categories
cat_clean = set(airlines['cleanliness']).difference(categories['cleanliness'])

# Find rows with that category
cat_clean_rows = airlines['cleanliness'].isin(cat_clean)

# Print rows with inconsistent category
print(airlines[cat_clean_rows])

# Print rows with consistent categories only
print(airlines[~cat_clean_rows])

# Inconsistent categories
#
# In this exercise, you'll be revisiting the airlines DataFrame from the previous exercises. As a reminder,
# the DataFrame contains flight metadata such as the airline, the destination, waiting times as well as answers to
# key questions regarding cleanliness, safety, and satisfaction on the San Francisco Airport.
#
# In this exercise, you will examine two categorical columns from this DataFrame, dest_region and dest_size
# respectively, assess how to address them and make sure that they are cleaned and ready for analysis. The pandas
# package has been imported as pd, and the airlines DataFrame is in your environment.

print('-------------------')
print(airlines['dest_region'].value_counts())
print(airlines['dest_size'].value_counts())

# Print unique values of both columns
print('-------------------')
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())

# Lower dest_region column and then replace "eur" with "europe"
airlines['dest_region'] = airlines['dest_region'].str.lower()
airlines['dest_region'] = airlines['dest_region'].replace({'eur': 'europe'})

# Remove white spaces from `dest_size`
airlines['dest_size'] = airlines['dest_size'].str.strip()

# Verify changes have been effected
print('-------------------')
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())

# Remapping categories
#
# To better understand survey respondents from San Francisco, you want to find out if there is a relationship
# between certain responses and the day of the week and wait time at the gate.
#
# The airlines DataFrame contains the day and wait_min columns, which are categorical and numerical respectively.
# The day column contains the exact day a flight took place, and wait_min contains the amount of minutes it took
# travelers to wait at the gate. To make your analysis easier, you want to create two new categorical variables:
#
# wait_type: 'short' for 0-60 min, 'medium' for 60-180 and long for 180+
# day_week: 'weekday' if day is in the weekday, 'weekend' if day is in the weekend.

# Create ranges for categories
# np.inf is used to denote infinity
label_ranges = [0, 60, 180, np.inf]
label_names = ['short', 'medium', 'long']

# Create wait_type column
airlines['wait_type'] = pd.cut(airlines['wait_min'], bins=label_ranges, labels=label_names)

# Create mappings and replace
mappings = {'Mon': 'weekday', 'Tue': 'weekday', 'Wed': 'weekday', 'Thu': 'weekday', 'Fri': 'weekday',
            'Sat': 'weekend', 'Sun': 'weekend'}

airlines['day_week'] = airlines['day'].replace(mappings)


# Removing titles and taking names
#
# While collecting survey respondent metadata in the airlines DataFrame, the full name of respondents was saved
# in the full_name column. However upon closer inspection, you found that a lot of the different names are prefixed
# by honorifics such as "Dr.", "Mr.", "Ms." and "Miss".
#
# Your ultimate objective is to create two new columns named first_name and last_name, containing the first and
# last names of respondents respectively. Before doing so however, you need to remove honorifics.
#

# Replace "Dr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace('Dr.', '')

# Replace "Mr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace('Mr.', '')

# Replace "Miss" with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace('Miss', '')

# Replace "Ms." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace('Ms.', '')

# Assert that full_name has no honorifics
assert airlines['full_name'].str.contains('Ms.|Mr.|Miss|Dr.').any() == False

# Keeping it descriptive
#
# To further understand travelers' experiences in the San Francisco Airport, the quality assurance department
# sent out a qualitative questionnaire to all travelers who gave the airport the worst score on all possible
# categories. The objective behind this questionnaire is to identify common patterns in what travelers are
# saying about the airport.
#
# Their response is stored in the survey_response column. Upon a closer look, you realized a few of the answers
# gave the shortest possible character amount without much substance. In this exercise, you will isolate the
# responses with a character count higher than 40 , and make sure your new DataFrame contains responses with 40
# characters or more using an assert statement.

# Store length of each row in survey_response column
resp_length = airlines['survey_response'].str.len()

# Find rows in airlines where resp_length > 40
airlines_survey = airlines[resp_length > 40]

# Assert minimum survey_response length is > 40
assert airlines_survey['survey_response'].str.len().min() > 40

# Print new survey_response column
print(airlines_survey['survey_response'])
