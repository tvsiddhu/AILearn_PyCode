import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 2. Making a count plot with a list
# In this exercise, we'll use a dataset that contains information about 227 countries. This dataset has lots of
# interesting information on each country, such as the country's birth rates, death rates, and its gross domestic
# product (GDP). GDP is the value of all the goods and services produced in a year, expressed as dollars per person.
#
# We've created a list of data from this dataset to get you started. birth_rate is a list that contains the birth
# rate of each country. The birth rate is the number of births per 1,000 people in a year.
# Read the data from the csv file
df = pd.read_csv('../../data/eda_data_sources/countries-of-the-world.csv')
gdp = df['GDP ($ per capita)']
phones = df['Phones (per 1000)']
percent_literate = df['Literacy (%)']

birth_rate = df['Birthrate']
region = df['Region']

# Create count plot with region on the y-axis
sns.countplot(y=region)
plt.show()

# 2. "Tidy" vs. "untidy" data
# Here, we have a sample dataset from a survey of children about their favorite animals. But
# can we use this dataset as-is with Seaborn? Let's use pandas to import the csv file with the data collected from
# the survey and determine whether it is tidy, which is essential to having it work well with Seaborn.

# Read the data from the csv file
df = pd.read_csv('../../data/eda_data_sources/animals.csv')

# Print the head of the data
print(df.head())

# 3. Making a count plot with a DataFrame
# In this exercise, we'll look at the responses to a survey sent out to young people. Our primary question here is:
# how many young people surveyed report being scared of spiders? Survey participants were asked to agree or disagree
# with the statement "I am afraid of spiders". Responses vary from 1 to 5, where 1 is "Strongly disagree" and 5 is
# "Strongly agree".
#

df = pd.read_csv('../../data/eda_data_sources/young-people-survey-responses.csv')
# Create a count plot with "Spiders" on the x-axis
sns.countplot(x='Spiders', data=df)

# Display the plot
plt.show()

# 5. Fill in the palette_colors dictionary to map the "Rural" location value to the color "green" and the "Urban"
# location value to the color "blue". Create a count plot with "school" on the x-axis using the student_data
# DataFrame. Add subgroups to the plot using "location" variable and use the palette_colors dictionary to make the
# location subgroups green and blue.

# Import student data
student_data = pd.read_csv('../../data/eda_data_sources/student-alcohol-consumption.csv')

# Create a dictionary mapping subgroup values to colors
palette_colors = {"Rural": "green", "Urban": "blue"}

# Create a count plot of school with location subgroups
sns.countplot(x='school', data=student_data, hue='location', palette=palette_colors)

# Display plot
plt.show()

# 14.Bar plots with percentages
# Let's continue exploring the responses to a survey sent out to young people. The
# variable "Interested in Math" is True if the person reported being interested or very interested in mathematics,
# and False otherwise. What percentage of young people report being interested in math, and does this vary based on
# gender? Let's use a bar plot to find out.
#
# As a reminder, we'll create a bar plot using the catplot() function, providing the name of categorical variable to
# put on the x-axis (x=____), the name of the quantitative variable to summarize on the y-axis (y=____), the pandas
# DataFrame to use (data=____), and the type of categorical plot (kind="bar").

# Use the survey_data DataFrame and sns.catplot() to create a bar plot with "Gender" on the x-axis and "Interested in
# Math" on the y-axis.

survey_data = pd.read_csv('../../data/eda_data_sources/young-people-survey-responses-math.csv')

# Create a bar plot of interest in math, separated by gender
sns.catplot(x="Gender", y="Interested in Math", data=survey_data, kind="bar")

# Show plot
plt.show()

# 15. Customizing bar plots
# In this exercise, we'll explore data from students in secondary school. The "study_time"
# variable records each student's reported weekly study time as one of the following categories: "<2 hours",
# "2 to 5 hours", "5 to 10 hours", or ">10 hours". Do students who report higher amounts of studying tend to get
# better final grades? Let's compare the average final grade among students in each category using a bar plot.

# Import student data
student_data = pd.read_csv('../../data/eda_data_sources/student-alcohol-consumption.csv')

# Create a bar plot of average final grade in each study category
sns.catplot(x="study_time", y="G3", data=student_data, kind="bar")

# Show plot
plt.show()

# Using the order parameter and the category_order list that is provided, rearrange the bars so that they are in
# order from the lowest study time to highest.
category_order = ["<2 hours", "2 to 5 hours", "5 to 10 hours", ">10 hours"]

# Create a bar plot of average final grade in each study category
sns.catplot(x="study_time", y="G3", data=student_data, kind="bar", order=category_order)

# Show plot
plt.show()

# Update the plot so that it no longer displays confidence intervals.
sns.catplot(x="study_time", y="G3", data=student_data, kind="bar", order=category_order, errorbar=None)

# Show plot
plt.show()