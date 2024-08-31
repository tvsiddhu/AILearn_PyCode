import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Making a scatter plot with lists
# In this exercise, we'll use a dataset that contains information about 227
# countries. This dataset has lots of interesting information on each country, such as the country's birth rates,
# death rates, and its gross domestic product (GDP). GDP is the value of all the goods and services produced in a
# year, expressed as dollars per person.
#
# We've created three lists of data from this dataset to get you started. gdp is a list that contains the value of
# GDP per country, expressed as dollars per person. phones is a list of the number of mobile phones per 1,000 people
# in that country. Finally, percent_literate is a list that contains the percent of each country's population that
# can read and write.

# Read the data from the csv file
df = pd.read_csv('../../data/eda_data_sources/countries-of-the-world.csv')
gdp = df['GDP ($ per capita)']
phones = df['Phones (per 1000)']
percent_literate = df['Literacy (%)']

# Create scatter plot with GDP on the x-axis and number of phones on the y-axis
sns.scatterplot(x=gdp, y=phones)
plt.show()

# Change the scatter plot so it displays the percent of the population that can read and write (percent_literate) on
# the y-axis.
sns.scatterplot(x=gdp, y=percent_literate)
plt.show()

# 4. Hue and scatter plots
# In the prior video, we learned how hue allows us to easily make subgroups within Seaborn plots. Let's try it out by
# exploring data from students in secondary school. We have a lot of information about each student like their age,
# where they live, their study habits and their extracurricular activities.
#
# For now, we'll look at the relationship between the number of absences they have in school and their final grade in
# the course, segmented by where the student lives (rural vs. urban area).

# Read the data from the csv file
student_data = pd.read_csv('../../data/eda_data_sources/student-alcohol-consumption.csv')

# Create a scatter plot of absences vs. final grade
sns.scatterplot(x='absences', y='G3', data=student_data, hue='location')

# Show plot
plt.show()

# Make "Rural" appear before "Urban"
sns.scatterplot(x='absences', y='G3', data=student_data, hue='location', hue_order=['Rural', 'Urban'])

# Show plot
plt.show()

# 6. Creating subplots with col and row We've seen in prior exercises that students with more absences ("absences")
# tend to have lower final grades ("G3"). Does this relationship hold regardless of how much time students study each
# week?
#
# To answer this, we'll look at the relationship between the number of absences that a student has in school and
# their final grade in the course, creating separate subplots based on each student's weekly study time ("study_time").

# Change to use relplot() instead of scatterplot()
sns.relplot(x="absences", y="G3", data=student_data, kind="scatter", col='study_time')

# Show plot
plt.show()

# Modify the code to create one scatter plot for each level of the variable "study_time", arranged in columns.
sns.relplot(x="absences", y="G3", data=student_data, kind="scatter", col='study_time',
            col_order=['<2 hours', '2 to 5 hours', '5 to 10 hours', '>10 hours'])

# Show plot
plt.show()

# Adapt your code to create one scatter plot for each level of a student's weekly study time, this time arranged in
# rows.
sns.relplot(x="absences", y="G3", data=student_data, kind="scatter", row='study_time')

# Show plot
plt.show()

# 7. Creating two-factor subplots Let's continue looking at the student_data dataset of students in secondary school.
# Here, we want to answer the following question: does a student's first semester grade ("G1") tend to correlate with
# their final grade ("G3")?
#
# There are many aspects of a student's life that could result in a higher or lower final grade in the class. For
# example, some students receive extra educational support from their school ("schoolsup") or from their family (
# "famsup"), which could result in higher grades. Let's try to control for these two factors by creating subplots
# based on whether the student received extra educational support from their school or family.

# Create a scatter plot of G1 vs. G3
sns.relplot(x="G1", y="G3", data=student_data, kind="scatter", col='schoolsup', col_order=['yes', 'no'], row='famsup',
            row_order=['yes', 'no'])

# Show plot
plt.show()

# Create column subplots based on whether the student received support from the school ("schoolsup"), ordered so that
# "yes" comes before "no".
sns.relplot(x="G1", y="G3", data=student_data, kind="scatter", col='schoolsup', col_order=['yes', 'no'])

# Show plot
plt.show()

# Add row subplots based on whether the student received support from the family ("famsup"), ordered so that "yes"
# comes before "no". This will result in subplots based on two factors.
sns.relplot(x="G1", y="G3", data=student_data, kind="scatter", col='schoolsup', col_order=['yes', 'no'], row='famsup',
            row_order=['yes', 'no'])

# Show plot
plt.show()

# 8 Changing the size of scatter plot points In this exercise, we'll explore Seaborn's mpg dataset, which contains
# one row per car model and includes information such as the year the car was made, the number of miles per gallon (
# "M.P.G.") it achieves, the power of its engine (measured in "horsepower"), and its country of origin.
#
# What is the relationship between the power of a car's engine ("horsepower") and its fuel efficiency ("mpg")? And
# how does this relationship vary by the number of cylinders ("cylinders") the car has? Let's find out.
#
# Let's continue to use relplot() instead of scatterplot() since it offers more flexibility.

# Use relplot() and the mpg DataFrame to create a scatter plot with "horsepower" on the x-axis and "mpg" on the
# y-axis. Vary the size of the points by the number of cylinders in the car ("cylinders").

mpg = pd.read_csv('../../data/eda_data_sources/mpg.csv')

# Create a scatter plot of horsepower vs. mpg
sns.relplot(x="horsepower", y="mpg", data=mpg, kind="scatter", size="cylinders")

# Show plot
plt.show()

# To make this plot easier to read, use the hue semantic to create subgroups based on the number of cylinders in the
# car.

# Create a scatter plot of horsepower vs. mpg
sns.relplot(x="horsepower", y="mpg", data=mpg, kind="scatter", size="cylinders", hue='cylinders')

# Show plot
plt.show()

# 9. Changing the style of scatter plot points Let's continue exploring Seaborn's mpg dataset by looking at the
# relationship between how fast a car can accelerate ("acceleration") and its fuel efficiency ("mpg"). Do these
# properties vary by country of origin ("origin")?
#
# Note that the "acceleration" variable is the time to accelerate from 0 to 60 miles per hour, in seconds. Higher
# values indicate slower acceleration.

# Create a scatter plot of acceleration vs. mpg
sns.relplot(x="acceleration", y="mpg", data=mpg, kind="scatter", style='origin', hue='origin')

# Show plot
plt.show()
