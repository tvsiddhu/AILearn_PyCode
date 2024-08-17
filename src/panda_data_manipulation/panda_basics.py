import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Inspecting a DataFrame When you get a new DataFrame to work with, the first thing you need to do is explore it and
# see what it contains. There are several useful methods and attributes for this.
#
# .head() returns the first few rows (the “head” of the DataFrame). .info() shows information on each of the columns,
# such as the data type and number of missing values. .shape returns the number of rows and columns of the DataFrame.
# .describe() calculates a few summary statistics for each column. homelessness is a DataFrame containing estimates
# of homelessness in each U.S. state in 2018. The individual column is the number of homeless individuals not part of
# a family with children. The family_members column is the number of homeless individuals part of a family with
# children. The state_pop column is the state's total population.

# Print the head of the homelessness data

pd.set_option('display.max_columns', 10)
homelessness = pd.read_csv('../../data/panda_data_manipulation_sources/homelessness.csv')
print(homelessness.head())

# Print information about homelessness
print(homelessness.info())

# Print the shape of homelessness
print(homelessness.shape)

# Print a description of homelessness
print(homelessness.describe())

# Parts of a DataFrame To better understand DataFrame objects, it's useful to know that they consist of three
# components, stored as attributes:
#
# .values: A two-dimensional NumPy array of values. .columns: An index of columns: the column names. .index: An index
# for the rows: either row numbers or row names. You can usually think of indexes as a list of strings or numbers,
# though the pandas Index data type allows for more sophisticated options. (These will be covered later in the course.)

# Print the values of homelessness
print(homelessness.values)

# Print the column index of homelessness
print(homelessness.columns)

# Print the row index of homelessness
print(homelessness.index)

# Sorting rows Finding interesting bits of data in a DataFrame is often easier if you change the order of the rows.
# You can sort the rows by passing a column name to .sort_values().
#
# In cases where rows have the same value (this is common if you sort on a categorical variable), you may wish to
# break the ties by sorting on another column. You can sort on multiple columns in this way by passing a list of
# column names.
#
# Sort on …	Syntax
# one column	df.sort_values("breed")
# multiple columns	df.sort_values(["breed", "weight_kg"])
# By combining .sort_values() with .head(), you can answer questions in the form, "What are the top cases where…?".

# Sort homelessness by individuals
homelessness_ind = homelessness.sort_values('individuals')

# Print the top few rows
print(homelessness_ind.head())

# Sort homelessness by descending family members
homelessness_fam = homelessness.sort_values('family_members', ascending=False)

# Print the top few rows
print(homelessness_fam.head())

# Sort homelessness by region, then descending family members
homelessness_reg_fam = homelessness.sort_values(['region', 'family_members'], ascending=[True, False])

# Print the top few rows
print(homelessness_reg_fam.head())

# Subsetting columns When working with data, you may not need all the variables in your dataset. Square-brackets
# ([]) can be used to select only the columns that matter to you in a DataFrame.
#
# To select only "col_a" of the DataFrame df, use df["col_a"]. To select "col_a" and "col_b" of df, use
# df[["col_a", "col_b"]].

# Select the individuals column
individuals = homelessness['individuals']

# Print the head of the result
print(individuals.head())

# Select the state and family_members columns
state_fam = homelessness[['state', 'family_members']]
print(state_fam.head())

# Select only the individuals and state columns, in that order
ind_state = homelessness[['individuals', 'state']]
print(ind_state.head())

# Subsetting rows A large part of data science is about finding which bits of your dataset are interesting. One of the
# simplest techniques for this is to find a subset of rows that match some criteria. This is sometimes known as
# filtering rows or selecting rows.
#
# There are many ways to subset a DataFrame, perhaps the most common is to use relational operators to return True or
# False for each row, then pass that inside square brackets.
#
# dogs[dogs["height_cm"] > 60] dogs[dogs["color"] == "tan"] You can filter for multiple conditions at once by using the
# "bitwise and" operator, &.
#
# dogs[(dogs["height_cm"] > 60) & (dogs["col_b"] == "tan")]

# Filter for rows where individuals is greater than 10000
ind_gt_10k = homelessness[homelessness['individuals'] > 10000]

# See the result
print(ind_gt_10k)

# Filter for rows where region is Mountain
mountain_reg = homelessness[homelessness['region'] == 'Mountain']

# See the result
print(mountain_reg)

# Filter for rows where family_members is less than 1000 and region is Pacific
fam_lt_1k_pac = homelessness[(homelessness['family_members'] < 1000) & (homelessness['region'] == 'Pacific')]
print(fam_lt_1k_pac)

# Subsetting rows by categorical variables Sub-setting data based on a categorical variable often involves using the
# "or" operator (|) to select rows from multiple categories. This can get tedious when you want all states in one of
# three different regions, for example. Instead, use the .isin() method, which will allow you to tackle this problem
# by writing one condition instead of three separate ones.
#
# colors = ["brown", "black", "tan"]
# condition = dogs["color"].isin(colors)
# dogs[condition]

# The Mojave Desert states are Nevada, Arizona, Utah, and California.

canu = ['Nevada', 'Arizona', 'Utah', 'California']
mojave_homelessness = homelessness[homelessness['state'].isin(canu)]
print(mojave_homelessness)

# Adding new columns You aren't stuck with just the data you are given. Instead, you can add new columns to a
# DataFrame. This has many names, such as transforming, mutating, and feature engineering.
#
# You can create new columns from scratch, but it is also common to derive them from other columns, for example,
# by adding columns together or by changing their units.
#
# homelessness is a DataFrame containing estimates of homelessness in each U.S. state in 2018. The individual column
# is the number of homeless individuals not part of a family with children. The family_members column is the number
# of homeless individuals part of a family with children. The state_pop column is the state's total population.

# Add total col as sum of individuals and family_members
homelessness['total'] = homelessness['individuals'] + homelessness['family_members']

# Add p_homeless col as proportion of total homeless population to the state population
homelessness['p_homeless'] = homelessness['total'] / homelessness['state_pop']

# See the result
print(homelessness)

# Combo-attack! You've seen the four most common types of data manipulation: sorting rows, sub-setting columns,
# subsetting rows, and adding new columns. In a real-life data analysis, you can mix and match these four
# manipulations to answer a multitude of questions.
#
# In this exercise, you'll answer the question, "Which state has the highest number of homeless individuals per 10,
# 000 people in the state?" Combine your new pandas skills to find out.

# Create indiv_per_10k col as homeless individuals per 10k state pop
homelessness['indiv_per_10k'] = 10000 * homelessness['individuals'] / homelessness['state_pop']

# Subset rows for indiv_per_10k greater than 20
high_homelessness = homelessness[homelessness['indiv_per_10k'] > 20]

# Sort high_homelessness by descending indiv_per_10k
high_homelessness_srt = high_homelessness.sort_values('indiv_per_10k', ascending=False)

# From high_homelessness_srt, select the state and indiv_per_10k cols
result = high_homelessness_srt[['state', 'indiv_per_10k']]
print(result)

# Mean and median Summary statistics are exactly what they sound like - they summarize many numbers in one statistic.
# For example, mean, median, minimum, maximum, and standard deviation are summary statistics. Calculating summary
# statistics allows you to get a better sense of your data, even if there's a lot of it.

# Print the head of the sales DataFrame
sales = pd.read_csv('../../data/panda_data_manipulation_sources/sales_subset.csv')
print(sales.head())

# Print the info about the sales DataFrame
print(sales.info())

# Print the mean of weekly_sales
print(sales['weekly_sales'].mean())

# Print the median of weekly_sales
print(sales['weekly_sales'].median())

# Summarizing dates Summary statistics can also be calculated on date columns that have values with the data type
# datetime64. Some summary statistics — like mean — don't make a ton of sense on dates, but others are super helpful,
# for example, minimum and maximum, which allow you to see what time range your data covers.

# Print the maximum of the date column
print(sales['date'].max())

# Print the minimum of the date column
print(sales['date'].min())


# 1. Efficient summaries
# While pandas and NumPy have tons of functions, sometimes, you may need a different function to summarize your data.
#
# The .agg() method allows you to apply your own custom functions to a DataFrame, as well as apply functions to more
# than one column of a DataFrame at once, making your aggregations super-efficient. For example,
#
# df['column'].agg(function) In the custom function for this exercise, "IQR" is short for inter-quartile range,
# which is the 75th percentile minus the 25th percentile. It's an alternative to standard deviation that is helpful
# if your data contains outliers.

# A custom IQR function
def iqr(column):
    """

    :param column:
    :return:
    """
    return column.quantile(0.75) - column.quantile(0.25)


# Print IQR of the temperature_c column
print(sales['temperature_c'].agg(iqr))

# In the custom function for this exercise, "IQR" is short for inter-quartile range, which is the 75th percentile
# minus the 25th percentile. It's an alternative to standard deviation that is helpful if your data contains outliers.

# Update to print IQR of temperature_c, fuel_price_usd_per_l, & unemployment
# Update the aggregation functions called by .agg(): include iqr and np.median in that order.

print(sales[["temperature_c", "fuel_price_usd_per_l", "unemployment"]].agg([iqr, "median"]))

# Cumulative statistics can also be helpful in tracking summary statistics over time. In this exercise,
# you'll calculate the cumulative sum and cumulative max of a department's weekly sales, which will allow you to
# identify what the total sales were so far as well as what the highest weekly sales were so far.
#

# Sort sales_1_1 by date
sales_1_1 = sales.sort_values('date', ascending=True)

# Get the cumulative sum of weekly_sales, add as cum_weekly_sales col
sales_1_1['cum_weekly_sales'] = sales_1_1['weekly_sales'].cumsum()

# Get the cumulative max of weekly_sales, add as cum_max_sales col
sales_1_1['cum_max_sales'] = sales_1_1['weekly_sales'].cummax()

# See the columns you calculated
print(sales_1_1[["date", "weekly_sales", "cum_weekly_sales", "cum_max_sales"]])

# Dropping duplicates Removing duplicates is an essential skill to get accurate counts because often, you don't want
# to count the same thing multiple times. In this exercise, you'll create some new DataFrames using unique values
# from sales.

# Drop duplicate store/type combinations
store_types = sales.drop_duplicates(subset=['store', 'type'])
print(store_types.head())

# Drop duplicate store/department combinations
store_depts = sales.drop_duplicates(subset=['store', 'department'])
print(store_depts.head())

# Subset the rows where is_holiday is True and drop duplicate dates
holiday_dates = sales[sales['is_holiday']].drop_duplicates(subset='date')

# Print date col of holiday_dates
print(holiday_dates['date'])

# 2. Counting categorical variables
# Counting is a great way to get an overview of your data and to spot curiosities that
# you might not notice otherwise. In this exercise, you'll count the number of each type of store and the number of
# each department number using the DataFrames you created in the previous exercise:
#
# # Drop duplicate store/type combinations
# store_types = sales.drop_duplicates(subset=["store", "type"])
#
# # Drop duplicate store/department combinations
# store_depts = sales.drop_duplicates(subset=["store", "department"])

# Count the number of stores of each type
store_counts = store_types['type'].value_counts()
print(store_counts)

# Count the proportion of stores of each type
store_props = store_types['type'].value_counts(normalize=True)
print(store_props)

# Count the number of stores of each department number and sort
dept_counts_sorted = store_depts['department'].value_counts(sort=True)
print(dept_counts_sorted)

# Get the proportion of stores in each department and sort
dept_props_sorted = store_depts['department'].value_counts(sort=True, normalize=True)
print(dept_props_sorted)

# 3. What percent of sales occurred at each store type?
# While .groupby() is useful, you can calculate grouped summary statistics without it.
#
# Walmart distinguishes three types of stores: "supercenters," "discount stores," and "neighborhood markets,
# " encoded in this dataset as type "A," "B," and "C." In this exercise, you'll calculate the total sales made at
# each store type, without using .groupby(). You can then use these numbers to see what proportion of Walmart's total
# sales were made at each type.

# Calc total weekly sales
sales_all = sales['weekly_sales'].sum()

# Subset for type A stores, calc total weekly sales
sales_A = sales[sales['type'] == 'A']['weekly_sales'].sum()

# Subset for type B stores, calc total weekly sales
sales_B = sales[sales['type'] == 'B']['weekly_sales'].sum()

# Subset for type C stores, calc total weekly sales
sales_C = sales[sales['type'] == 'C']['weekly_sales'].sum()

# Get proportion for each type
sales_propn_by_type = [sales_A, sales_B, sales_C] / sales_all
print(sales_propn_by_type)

# Calculations with .groupby() The .groupby() method makes life much easier. In this exercise, you'll perform the
# same calculations as last time, except you'll use the .groupby() method. You'll also perform calculations on data
# grouped by two variables to see if sales differ by store type depending on if it's a holiday week or not.

# Group by type; calc total weekly sales
sales_by_type = sales.groupby('type')['weekly_sales'].sum()

# Get proportion for each type
sales_propn_by_type = sales_by_type / sum(sales_by_type)
print(sales_propn_by_type)

# Group sales by "type" and "is_holiday", take the sum of weekly_sales, and store as sales_by_type_is_holiday.
sales_by_type_is_holiday = sales.groupby(['type', 'is_holiday'])['weekly_sales'].sum()
print(sales_by_type_is_holiday)

# Group by type and is_holiday; calc total weekly sales
sales_by_type_is_holiday = sales.groupby(['type', 'is_holiday'])['weekly_sales'].sum()
print(sales_by_type_is_holiday)

# 4. Multiple grouped summaries
# Earlier in this chapter, you saw that the .agg() method is useful to compute multiple
# statistics on multiple variables. It also works with grouped data. NumPy, which is imported as np,
# has many different summary statistics functions, including: np.min, np.max, np.mean, and np.median.


# For each store type, aggregate weekly_sales: get min, max, mean, and median
sales_stats = sales.groupby('type')['weekly_sales'].agg(["min", "max", "mean", "median"])

# Print sales_stats
print(sales_stats)

# For each store type, aggregate unemployment and fuel_price_usd_per_l: get min, max, mean, and median
unemp_fuel_stats = sales.groupby('type')[['unemployment', 'fuel_price_usd_per_l']].agg(["min", "max", "mean", "median"])

# Print unemp_fuel_stats
print(unemp_fuel_stats)

# 5. Pivoting on one variable
# Pivot tables are the standard way of aggregating data in spreadsheets.
#
# In pandas, pivot tables are essentially another way of performing grouped calculations. That is, the .pivot_table()
# method is an alternative to .groupby().
#
# In this exercise, you'll perform calculations using .pivot_table() to replicate the calculations you performed in
# the last lesson using .groupby().

# Pivot for mean weekly_sales for each store type
mean_sales_by_type = sales.pivot_table(index='type', values='weekly_sales')

# Print mean_sales_by_type
print(mean_sales_by_type)

# Get the mean and median (using NumPy functions) of weekly_sales by type using .pivot_table() and store as
# mean_med_sales_by_type.

# Pivot for mean and median weekly_sales for each store type
mean_med_sales_by_type = sales.pivot_table(index='type', values='weekly_sales', aggfunc=["mean", "median"])

# Print mean_med_sales_by_type
print(mean_med_sales_by_type)

# Get the mean of weekly_sales by type and is_holiday using .pivot_table() and store as mean_sales_by_type_holiday.

# Pivot for mean weekly_sales by store type and holiday
mean_sales_by_type_holiday = sales.pivot_table(index='type', values='weekly_sales', columns='is_holiday')

# Print mean_sales_by_type_holiday
print(mean_sales_by_type_holiday)

# 6. Fill in missing values and sum values with pivot tables
# The .pivot_table() method has several useful arguments, including fill_value and margins.
#
# fill_value replaces missing values with a real value (known as imputation). What to replace missing values with is a
# topic big enough to have its own course (Dealing with Missing Data in Python), but the simplest thing to do is to
# substitute a dummy value.
#
# margins is a shortcut for when you pivoted by two variables, but also wanted to pivot by each of those variables
# separately: it gives the row and column totals of the pivot table contents.
#
# In this exercise, you'll practice using these arguments to up your pivot table skills, which will help you crunch
# numbers more efficiently!

# Print mean weekly_sales by department and type; fill missing values with 0
print(sales.pivot_table(index='department', columns='type', values='weekly_sales', fill_value=0))

# Print the mean weekly_sales by department and type, filling in any missing values with 0 and summing all rows and
# columns.

# Print the mean weekly_sales by department and type; fill missing values with 0s; sum all rows and cols
print(sales.pivot_table(index='department', columns='type', values='weekly_sales', fill_value=0, margins=True))

# 7. Setting and removing indexes
# pandas allows you to designate columns as an index. This enables cleaner code when
# taking subsets (as well as providing more efficient lookup under some circumstances).
#
# In this chapter, you'll be exploring temperatures, a DataFrame of average temperatures in cities around the world.

# Look at temperatures
temperatures = pd.read_csv('../../data/panda_data_manipulation_sources/temperatures.csv', parse_dates=['date'],
                           encoding='utf-8-sig')
print(temperatures)

# Index temperatures by city
temperatures_ind = temperatures.set_index('city')

# Look at temperatures_ind
print(temperatures_ind)

# Reset the temperature_ind index, keeping its contents
print(temperatures_ind.reset_index())

# Reset the temperature_ind index, dropping its contents
print(temperatures_ind.reset_index(drop=True))

# 8. Subsetting with .loc[]
# The killer feature for indexes is .loc[]: a subsetting method that accepts index values. When you pass it a single
# argument, it will take a subset of rows.
#
# The code for subsetting using .loc[] can be easier to read than standard square bracket subsetting, which can make
# your code less burdensome to maintain.

# Make a list of cities to subset on
cities = ['Moscow', 'Saint Petersburg']

# Subset temperatures using square brackets
print(temperatures[temperatures['city'].isin(cities)])

# Subset temperatures_ind using .loc[]
print(temperatures_ind.loc[cities])

# 9. Setting multi-level indexes
# Indexes can also be made out of multiple columns, forming a multi-level index (sometimes called a hierarchical index).
# There is a trade-off to using these.
#
# The benefit is that multi-level indexes make it more natural to reason about nested categorical variables. For
# example, in a clinical trial you might have control and treatment groups. Then each test subject belongs to one or
# other group, and we can say that test subject is nested inside treatment group.
#
# The main downside is that the code for manipulating indexes is more complex. In this exercise, you'll practice
# setting and removing multi-level indexes.

# Index temperatures by country & city
temperatures_ind = temperatures.set_index(['country', 'city'])

# List of tuples: Brazil, Rio De Janeiro & Pakistan, Lahore
rows_to_keep = [('Brazil', 'Rio De Janeiro'), ('Pakistan', 'Lahore')]

# Subset for rows to keep
print(temperatures_ind.loc[rows_to_keep])

# 10. Sorting by index values
# Previously, you changed the order of the rows in a DataFrame by calling .sort_values(). It's also useful to be able to
# sort by elements in the index. For this, you need to use .sort_index().

# Sort temperatures_ind by index values
print(temperatures_ind.sort_index())

# Sort temperatures_ind by index values at the city level
print(temperatures_ind.sort_index(level='city'))

# Sort temperatures_ind by country then descending city
print(temperatures_ind.sort_index(level=['country', 'city'], ascending=[True, False]))

# 11. Slicing index values
# Slicing lets you select consecutive elements of an object using first:last syntax.
# DataFrames can be sliced by index values or by row/column number; we'll start with the first case. This involves
# slicing inside the .loc[] method.
#
# Compared to slicing lists, there are a few things to remember.
#
# You can only slice an index if the index is sorted (using .sort_index()).
# To slice at the outer level, first and last can be strings.
# To slice at inner levels, first and last should be tuples.
# If you pass a single slice to .loc[], it will slice the rows.

# Sort the index of temperatures_ind
temperatures_srt = temperatures_ind.sort_index()

# Subset rows from Pakistan to Russia
print(temperatures_srt.loc['Pakistan':'Russia'])

# Try to subset rows from Lahore to Moscow
print(temperatures_srt.loc['Lahore':'Moscow'])

# Subset rows from Pakistan, Lahore to Russia, Moscow
print(temperatures_srt.loc[('Pakistan', 'Lahore'):('Russia', 'Moscow')])

# 12. Slicing in both directions
# You've seen slicing DataFrames by rows and by columns, but since DataFrames are two-dimensional objects, it is
# often natural to slice both dimensions at once. That is, by passing two arguments to .loc[], you can subset by rows
# and columns in one go.

# Subset rows from India, Hyderabad to Iraq, Baghdad
print(temperatures_srt.loc[('India', 'Hyderabad'):('Iraq', 'Baghdad')])

# Subset columns from date to avg_temp_c
print(temperatures_srt.loc[:, 'date':'avg_temp_c'])

# Subset in both directions at once
print(temperatures_srt.loc[('India', 'Hyderabad'):('Iraq', 'Baghdad'), 'date':'avg_temp_c'])

# 13. Slicing time series
# Slicing is particularly useful for time series since it's a common thing to want to filter for data within a date
# range. Add the date column to the index, then use .loc[] to perform the subsetting. The important thing to remember
# is to keep your dates in ISO 8601 format, that is, yyyy-mm-dd for year-month-day. "yyyy-mm" for year-month, or "yyyy"
# for year.

# Use Boolean conditions to subset temperatures for rows in 2010 and 2011
temperatures_bool = temperatures[(temperatures['date'] >= '2010-01-01') & (temperatures['date'] <= '2011-12-31')]
print(temperatures_bool)

# Set date as an index and sort the index
temperatures_ind = temperatures.set_index('date').sort_index()

# Use .loc[] to subset temperatures_ind for rows in 2010 and 2011
print(temperatures_ind.loc['2010':'2011'])

# Use .loc[] to subset temperatures_ind for rows from Aug 2010 to Feb 2011
print(temperatures_ind.loc['2010-08':'2011-02'])

# 14. Subsetting by row/column number
# The most common ways to subset rows are the ways we've previously discussed:
# using a Boolean condition or by index labels. However, it is also occasionally useful to pass row numbers directly.
# This is done using .iloc[], and like .loc[], it can take two arguments to let you subset by rows and columns.

# Get 23rd row, 2nd column (index 22, 1)
print(temperatures.iloc[22, 1])

# Use slicing to get the first 5 rows
print(temperatures.iloc[:5])

# Use slicing to get columns 3 to 4
print(temperatures.iloc[:, 2:4])

# Use slicing in both directions at once
print(temperatures.iloc[:5, 2:4])

# 15. Pivot temperature by city and year
# It's interesting to see how temperatures for each city change over
# time—looking at every month results in a big table, which can be tricky to reason about. Instead, let's look at how
# temperatures change by year.
#
# You can access the components of a date (year, month and day) using code of the form dataframe[
# "column"].dt.component. For example, the month component is dataframe["column"].dt.month, and the year component is
# dataframe["column"].dt.year.
#
# Once you have the year column, you can create a pivot table with the data aggregated by city and year, which you'll
# explore in the coming exercises.

# Add a year column to temperatures
temperatures['year'] = temperatures['date'].dt.year

# Pivot avg_temp_c by city and year
temp_by_country_city_vs_year = temperatures.pivot_table('avg_temp_c', index=['country', 'city'], columns='year')

# See the result
print(temp_by_country_city_vs_year)

# 16. Subsetting pivot tables
# A pivot table is just a DataFrame with sorted indexes, so the techniques you have learned already can be used to
# subset them. In particular, the .loc[] + slicing combination is often helpful.

# Subset for Egypt to India
print(temp_by_country_city_vs_year.loc['Egypt':'India'])

# Subset for Egypt, Cairo to India, Delhi
print(temp_by_country_city_vs_year.loc[('Egypt', 'Cairo'):('India', 'Delhi')])

# Subset in both directions at once
print(temp_by_country_city_vs_year.loc[('Egypt', 'Cairo'):('India', 'Delhi'), '2005':'2010'])

# 17. Calculating on a pivot table
# Pivot tables are filled with summary statistics, but they are only a first step to
# finding something insightful. Often you'll need to perform further calculations on them. A common thing to do is to
# find the rows or columns where the highest or lowest value occurs.
#
# Recall from Chapter 1 that you can easily subset a Series or DataFrame to find rows of interest using a logical
# condition inside square brackets. For example: series[series > value].

# Get the worldwide mean temp by year
mean_temp_by_year = temp_by_country_city_vs_year.mean()

# Find the year that had the highest mean temp
print(mean_temp_by_year[mean_temp_by_year == mean_temp_by_year.max()])

# Get the mean temp by city
mean_temp_by_city = temp_by_country_city_vs_year.mean(axis='columns')

# Find the city that had the lowest mean temp
print(mean_temp_by_city[mean_temp_by_city == mean_temp_by_city.min()])

# Creating and Visualizing DataFrames

# 18. Which avocado size is most popular?
# Avocados are increasingly popular and delicious in guacamole and on toast.
# The Hass Avocado Board keeps track of avocado supply and demand across the USA, including the sales of three
# different sizes of avocado. In this exercise, you'll use a bar plot to figure out which size is the most popular.
#
# Bar plots are great for revealing relationships between categorical (size) and numeric (number sold) variables,
# but you'll often have to manipulate your data first in order to get the numbers you need for plotting.

# Look at the first few rows of data
avocados = pd.read_csv('../../data/panda_data_manipulation_sources/avocados.csv')
print(avocados.head())

# Get the total number of avocados sold of each size
nb_sold_by_size = avocados.groupby('size')['nb_sold'].sum()

# Create a bar plot of the number of avocados sold by size
nb_sold_by_size.plot(kind='bar')

# Show the plot
plt.show()

# 19. Changes in sales over time
# Line plots are designed to visualize the relationship between two numeric variables, where each variable is a
# different dimension of the data. Because the x-axis is a time dimension, line plots can be used to show how a
# numeric variable changes over time.

# Get the total number of avocados sold on each date
nb_sold_by_date = avocados.groupby('date')['nb_sold'].sum()

# Create a line plot of the number of avocados sold by date
nb_sold_by_date.plot(kind='line')

# Show the plot
plt.show()

# 20. Avocado supply and demand
# Scatter plots are ideal for visualizing relationships between numerical variables. In this exercise, you'll compare
# the number of avocados sold to average price and see if they're at all related. If they're related, you may be able
# to use one number to predict the other.

# Scatter plot of nb_sold vs avg_price with title
avocados.plot(x='nb_sold', y='avg_price', kind='scatter', title='Number of avocados sold vs. average price')

# Show the plot
plt.show()

# 21. Price of conventional vs. organic avocados
# Creating multiple plots for different subsets of data allows you to
# compare groups. In this exercise, you'll create multiple histograms to compare the prices of conventional and
# organic avocados.

# Histogram of conventional avg_price
avocados[avocados['type'] == 'conventional']['avg_price'].hist()

# Histogram of organic avg_price
avocados[avocados['type'] == 'organic']['avg_price'].hist()

# Add a legend
plt.legend(['conventional', 'organic'])

# Show the plot
plt.show()

# Modify your code to adjust the transparency of both histograms to 0.5 to see how much overlap there is between the
# two distributions.

# Modify histogram transparency to 0.5
avocados[avocados['type'] == 'conventional']['avg_price'].hist(alpha=0.5)

# Modify histogram transparency to 0.5
avocados[avocados['type'] == 'organic']['avg_price'].hist(alpha=0.5)

# Add a legend
plt.legend(['conventional', 'organic'])

# Show the plot
plt.show()

# Modify your code to use 20 bins in both histograms.

# Modify bins to 20
avocados[avocados['type'] == 'conventional']['avg_price'].hist(alpha=0.5, bins=20)

# Modify bins to 20
avocados[avocados['type'] == 'organic']['avg_price'].hist(alpha=0.5, bins=20)

# Add a legend
plt.legend(['conventional', 'organic'])

# Show the plot
plt.show()

# 22. Finding missing values.
# Missing values are everywhere, and you don't want them interfering with your work. Some
# functions ignore missing data by default, but that's not always the behavior you might want. Some functions can't
# handle missing values at all, so these values need to be taken care of before you can use them. If you don't know
# where your missing values are, or if they exist, you could make mistakes in your analysis. In this exercise,
# you'll determine if there are missing values in the dataset, and if so, how many.

avocados_2016 = pd.read_csv('../../data/panda_data_manipulation_sources/avocados_2016.csv')
print(avocados_2016.head())

# Check individual values for missing values
print(avocados_2016.isna())

# Check each column for missing values
print(avocados_2016.isna().any())

# Bar plot of missing values by variable
avocados_2016.isna().sum().plot(kind='bar')

# Show plot
plt.show()

# 23. Removing missing values
# Now that you know there are some missing values in your DataFrame, you have a few
# options to deal with them. One way is to remove them from the dataset completely. In this exercise, you'll remove
# missing values by removing all rows that contain missing values.

# Remove rows with missing values
avocados_complete = avocados_2016.dropna()

# Check if any columns contain missing values
print(avocados_complete.isna().any())

# 24. Replacing missing values
# Another way of handling missing values is to replace them all with the same value. For
# numerical variables, one option is to replace values with 0— you'll do this here. However, when you replace missing
# values, you make assumptions about what a missing value means. In this case, you will assume that a missing number
# sold means that no sales for that avocado type were made that week.
#
# In this exercise, you'll see how replacing missing values can affect the distribution of a variable using
# histograms. You can plot histograms for multiple variables at a time as follows:
#
# dogs[["height_cm", "weight_kg"]].hist()

# List the columns with missing values
cols_with_missing = ["small_sold", "large_sold", "xl_sold"]

# Create histograms showing the distributions of small_sold, large_sold, and xl_sold
avocados_2016[cols_with_missing].hist()

# Show the plot
plt.show()

# Replace the missing values of avocados_2016 with 0s and store the result as avocados_filled.
# Create a histogram of the cols_with_missing columns of avocados_filled.

# Fill in missing values with 0
avocados_filled = avocados_2016.fillna(0)

# Create histograms of the filled columns
avocados_filled[cols_with_missing].hist()

# Show the plot
plt.show()

# 25. List of dictionaries
# You recently got some new avocado data from 2019 that you'd like to put in a DataFrame using the list of
# dictionaries method. Remember that with this method, you go through the data row by row.

# Create a list of dictionaries with new data
avocados_list = [
    {'date': '2019-11-03', 'small_sold': 10376832, 'large_sold': 7835071},
    {'date': '2019-11-10', 'small_sold': 10717154, 'large_sold': 8561348}
]

# Convert list into DataFrame
avocados_2019 = pd.DataFrame(avocados_list)

# Print the new DataFrame
print(avocados_2019)

# 26. Dictionary of lists
# Some more data just came in! This time, you'll use the dictionary of lists method, parsing the data column by column.

# Create a dictionary of lists with new data
avocados_dict = {
    'date': ['2019-11-17', '2019-12-01'],
    'small_sold': [10859987, 9291631],
    'large_sold': [7674135, 6238096]
}

# Convert dictionary into DataFrame
avocados_2019 = pd.DataFrame(avocados_dict)

# Print the new DataFrame
print(avocados_2019)

# 27. CSV to DataFrame
# You work for an airline, and your manager has asked you to do a competitive analysis and see
# how often passengers flying on other airlines are involuntarily bumped from their flights. You got a CSV file (
# airline_bumping.csv) from the Department of Transportation containing data on passengers that were involuntarily
# denied boarding in 2016 and 2017, but it doesn't have the exact numbers you want. In order to figure this out,
# you'll need to get the CSV into a pandas DataFrame and do some manipulation!

# Read CSV as DataFrame called airline_bumping
airline_bumping = pd.read_csv('../../data/panda_data_manipulation_sources/airline_bumping.csv')

# Take a look at the DataFrame
print(airline_bumping.head())

# For each airline group, select the nb_bumped, and total_passengers columns, and calculate the sum (for both years).
# Store this as airline_totals.

# For each airline group, select the nb_bumped and total_passengers columns and sum them
airline_totals = airline_bumping.groupby('airline')[['nb_bumped', 'total_passengers']].sum()

# Create a new column of airline_totals called bumps_per_10k, which is the number of passengers bumped per 10,000
# passengers in 2016 and 2017.

# Create a new column of airline_totals called bumps_per_10k for each airline
airline_totals['bumps_per_10k'] = airline_totals['nb_bumped'] / airline_totals['total_passengers'] * 10000

# Print airline_totals
print(airline_totals)

# 28. DataFrame to CSV
# You're almost there! To make things easier to read, you'll need to sort the data and export it
# to CSV so that your colleagues can read it.

# Create airline_totals_sorted
airline_totals_sorted = airline_totals.sort_values('bumps_per_10k', ascending=False)

# Print airline_totals_sorted
print(airline_totals_sorted)

# Save as airline_totals_sorted.csv
airline_totals_sorted.to_csv('../../data/panda_data_manipulation_sources/airline_totals_sorted.csv')
