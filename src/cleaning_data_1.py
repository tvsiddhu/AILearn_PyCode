import pandas as pd
import datetime as dt

ride_sharing = pd.read_csv('../data/cleaning_data_sources/ride_sharing_tire_sizes.csv')

# Print the information of ride_sharing
print(ride_sharing.info())

# Print summary statistics of user_type column
print(ride_sharing['user_type'].describe())

# Convert user_type from integer to category
ride_sharing['user_type_cat'] = ride_sharing['user_type'].astype('category')

# Write an assert statement confirming the change
assert ride_sharing['user_type_cat'].dtype == 'category'

# Print new summary statistics
print(ride_sharing['user_type_cat'].describe())

# Summing strings and concatenating numbers In the previous exercise, you were able to identify that category is the
# correct data type for user_type and convert it in order to extract relevant statistical summaries that shed light
# on the distribution of user_type.
#
# Another common data type problem is importing what should be numerical values as strings, as mathematical
# operations such as summing and multiplication lead to string concatenation, not numerical outputs.
#
# In this exercise, you'll be converting the string column duration to the type int. Before that however,
# you will need to make sure to strip "minutes" from the column in order to make sure pandas reads it as numerical.

# Strip duration of minutes
ride_sharing['duration_trim'] = ride_sharing['duration'].str.strip('minutes')

# Convert duration to integer
ride_sharing['duration_time'] = ride_sharing['duration_trim'].astype('int')

# Write an assert statement making sure of conversion
assert ride_sharing['duration_time'].dtype == 'int'

# Print formed columns and calculate average ride duration
print("-------------------")
print(ride_sharing[['duration', 'duration_trim', 'duration_time']])
print(ride_sharing['duration_time'].mean())

# Tire size constraints
# In this lesson, you're going to build on top of the work you've been doing with the ride_sharing data.
# You'll be working with the tire_sizes column which contains data on each bike's tire size.
#
# Bicycle tire sizes could be either 26″, 27″ or 29″ and are here correctly stored as a categorical value.
# In an effort to cut maintenance costs, the ride-sharing provider decided to set the maximum tire size to be 27″.
# Any bikes with tires larger than 27″ will be replaced.
#
# In this exercise, you will make sure the tire_sizes column has the correct range by first converting it to an integer.
# Note that the tire_sizes column has been loaded as a string.

# Convert tire_sizes to integer
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('int')

# Set all values above 27 to 27
ride_sharing.loc[ride_sharing['tire_sizes'] > 27, 'tire_sizes'] = 27

# Reconvert tire_sizes back to categorical
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('category')

# Print tire size description
print(ride_sharing['tire_sizes'].describe())

# Back to the future
# A new update has been issued that changes how the tire size information is stored. This new update will enable
# storing the tire sizes as integers and half integers. You realize that you could have avoided some data quality
# issues if you had this update earlier.


# Convert ride_date to date
ride_sharing['ride_dt'] = pd.to_datetime(ride_sharing['ride_date']).dt.date

# Save today's date
today = dt.date.today()

# Set all in the future to today's date
ride_sharing.loc[ride_sharing['ride_dt'] > today, 'ride_dt'] = today

# Print maximum of ride_dt column
print(ride_sharing['ride_dt'].max())

# Finding duplicates
# A new update to the data pipeline feeding into the ride_sharing DataFrame has added the ride_id column,
# which represents a unique identifier for each ride.
#
# The update however coincided with radically shorter average ride duration times and irregular user birthdate
# set in the future. Most importantly, the number of rides taken has increased by 20% overnight, leading you to think
# there might be both complete and incomplete duplicates in the ride_sharing DataFrame.
#
# In this exercise, you will confirm this suspicion by finding those duplicates.

# Find duplicates
duplicates = ride_sharing.duplicated(subset='ride_id', keep=False)

# Sort your duplicated rides
duplicated_rides = ride_sharing[duplicates].sort_values('ride_id')

# Print relevant columns of duplicated_rides
print(duplicated_rides[['ride_id', 'duration', 'user_birth_year']])

# Treating duplicates
# In the last exercise, you were able to verify that the new update feeding into the ride-sharing DataFrame contains
# a bug generating both complete and incomplete duplicated rows for some values of the ride_id column, with occasional
# discrepant values for the user_birth_year and duration columns.
#
# In this exercise, you will be treating those duplicated rows by first dropping complete duplicates,
# and then merging the incomplete duplicate rows into one while keeping the average duration, and the minimum
# user_birth_year for each set of incomplete duplicate rows.

# Drop complete duplicates from ride_sharing
ride_dup = ride_sharing.drop_duplicates()

# Create statistics dictionary for aggregation function
statistics = {'user_birth_year': 'min', 'duration': 'mean'}

# Group by ride_id and compute new statistics
ride_unique = ride_dup.groupby('ride_id').agg(statistics).reset_index()

# Find duplicated values again
duplicates = ride_unique.duplicated(subset='ride_id', keep=False)
duplicated_rides = ride_unique[duplicates == True]

# Assert duplicates are processed
assert duplicated_rides.shape[0] == 0
