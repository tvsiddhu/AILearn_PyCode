import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Import student data
student_data = pd.read_csv('../../../data/6. eda_data_sources/student-alcohol-consumption.csv')

# 16. Customizing point plots
# Let's continue to look at data from students in secondary school, this time using a
# point plot to answer the question: does the quality of the student's family relationship influence the number of
# absences the student has in school? Here, we'll use the "famrel" variable, which describes the quality of a
# student's family relationship from 1 (very bad) to 5 (very good).
#
# As a reminder, to create a point plot, use the catplot() function and specify the name of the categorical variable
# to put on the x-axis (x=____), the name of the quantitative variable to summarize on the y-axis (y=____),
# the pandas DataFrame to use (data=____), and the type of categorical plot (kind="point").
#
# Create a point plot of family relationship ("famrel") versus number of absences ("absences")
sns.catplot(x="famrel", y="absences", data=student_data, kind="point")

# Show plot
plt.show()

# Add "caps" to the end of the confidence intervals with size 0.2.
sns.catplot(x="famrel", y="absences", data=student_data, kind="point", capsize=0.2)

# Show plot
plt.show()

# Remove the lines joining the points in each category.
sns.catplot(x="famrel", y="absences", data=student_data, kind="point", linestyle="none")

# Show plot
plt.show()

# Point plots with subgroups
# Let's continue exploring the dataset of students in secondary school.
# This time, we'll ask: is being in a romantic relationship associated with higher or lower
# school attendance? And does this association differ by which school the students attend?
# Let's find out using a point plot.

# Create a point plot with subgroups based on school support
sns.catplot(x="romantic", y="absences", data=student_data, kind="point", hue="school", errorbar=None)

# Show plot
plt.show()

# Since there may be outliers of students with many absences, use the median function that we've imported from numpy
# to display the median number of absences instead of the average.

sns.catplot(x="romantic", y="absences", data=student_data, kind="point", hue="school", estimator=np.median)

# Show plot
plt.show()

# Changing style and palette
# Let's return to our dataset containing the results of a survey given to young people
# about their habits and preferences. We've provided the code to create a count plot of their responses to the
# question "How often do you listen to your parents' advice?". Now let's change the style and palette to make this
# plot easier to interpret.
survey_data = pd.read_csv('../../../data/6. eda_data_sources/young-people-survey-responses-parents.csv')

# Set the style to "whitegrid" to help the audience determine the number of responses in each category.
sns.set_style("whitegrid")

# Set the color palette to the sequential palette named "Purples".
sns.set_palette("Purples")

# Create a count plot of survey responses
category_order = ["Never", "Rarely", "Sometimes", "Often", "Always"]
sns.catplot(x="Parents Advice", data=survey_data, kind="count", order=category_order)

# Show plot
plt.show()

# change the color palette to the diverging palette named "RdBu".
sns.set_palette("Spectral")
sns.catplot(x="Parents Advice", data=survey_data, kind="count", order=category_order)

# Show plot
plt.show()

# Changing the scale
# In this exercise, we'll continue to look at the dataset containing responses from a survey of young people.
# Does the percentage of people reporting that they feel lonely vary depending on how many siblings they have?
# Let's find out using a bar plot, while also adjusting the style of the plot.
survey_data = pd.read_csv('../../../data/6. eda_data_sources/young-people-survey-responses-siblings.csv')

# Set the scale ("context") to "paper", which is the smallest of the scale options.
sns.set_context("paper")

# Create a bar plot
sns.catplot(x="Number of Siblings", y="Feels Lonely", data=survey_data, kind="bar")

# Show plot
plt.show()

# Change the context to "notebook" to increase the scale.
sns.set_context("notebook")
sns.catplot(x="Number of Siblings", y="Feels Lonely", data=survey_data, kind="bar")

# Show plot
plt.show()

# Change the context to "talk" to increase the scale.
sns.set_context("talk")
sns.catplot(x="Number of Siblings", y="Feels Lonely", data=survey_data, kind="bar")

# Show plot
plt.show()

# Change the context to "poster", which is the largest scale available.
sns.set_context("poster")
sns.catplot(x="Number of Siblings", y="Feels Lonely", data=survey_data, kind="bar")

# Show plot
plt.show()

# Using a custom palette
# So far, we've looked at several things in the dataset of survey responses from young people, including their
# habits and preferences. Now, we'll return to this dataset and use a custom color palette to create a count plot
# that shows the number of people who live in rural areas, suburbs, and cities.
survey_data = pd.read_csv('../../../data/6. eda_data_sources/young-people-survey-responses-location.csv')

# Set the style to "darkgrid"
sns.set_style("darkgrid")

# Set a custom color palette
sns.set_palette(["#39A7D0", "#36ADA4"])

# Create a count plot of survey responses
sns.catplot(x="Village - town", data=survey_data, kind="count")

# Show plot
plt.show()

# FacetGrids vs. AxesSubplots
# In the recent lesson, we learned that Seaborn plot functions create two different types of objects:
# FacetGrid objects and AxesSubplot objects. The method for adding a title to your plot will differ depending on the
# type of object it is.
#
# In the code provided, we've used relplot() with the miles per gallon dataset to create a scatter plot showing the
# relationship between a car's weight and its horsepower. This scatter plot is assigned to the variable name g. Let's
# add a title to this plot.
mpg = pd.read_csv('../../../data/6. eda_data_sources/mpg.csv')

# Create a scatter plot of weight vs. horsepower
g = sns.relplot(x="weight", y="horsepower", data=mpg, kind="scatter")

# Identify the type of plot
type_of_g = type(g)

# Print the type of plot
print("Type of plot: ", type_of_g)

# Add a title "Car Weight vs. Horsepower"
g.fig.suptitle("Car Weight vs. Horsepower", y=1.03)

# Show plot
plt.show()

# Adding a title and axis labels
# Let's continue to look at the miles per gallon dataset. This time we'll create a
# line plot to answer the question: How does the average miles per gallon achieved by cars change over time for each
# of the three places of origin? To improve the readability of this plot, we'll add a title and more informative axis
# labels.
#
# In the code provided, we create the line plot using the lineplot() function. Note that lineplot() does not support
# the creation of subplots, so it returns an AxesSubplot object instead of an FacetGrid object.

# Add the following title to the plot: "Average MPG Over Time".

mpg_mean = pd.read_csv('../../../data/6. eda_data_sources/mpg_mean.csv')

sns.set_context("notebook")
sns.set_palette("bright")

g = sns.lineplot(x="model_year", y="mpg_mean",
                 data=mpg_mean,
                 hue="origin")

plt.figure(figsize=(10, 6))
g.set_title("Average MPG Over Time")

# Label the x-axis as "Car Model Year".
g.set(xlabel="Car Model Year")
# Label the y-axis as "Average MPG".
g.set(ylabel="Average MPG")

sns.lineplot(x="model_year", y="mpg", data=mpg, hue="origin", errorbar=None)

# Show plot
plt.show()

# Rotating x-tick labels
# In this exercise, we'll continue looking at the miles per gallon dataset. In the code
# provided, we create a point plot that displays the average acceleration for cars in each of the three places of
# origin. Note that the "acceleration" variable is the time to accelerate from 0 to 60 miles per hour, in seconds.
# Higher values indicate slower acceleration.
#
# Let's use this plot to practice rotating the x-tick labels. Recall that the function to rotate x-tick labels is a
# standalone Matplotlib function and not a function applied to the plot object itself.

# Create a point plot of acceleration vs. origin
sns.catplot(x="origin", y="acceleration", data=mpg, kind="point", capsize=0.1, linestyle=None)

# Rotate x-tick labels
plt.xticks(rotation=90)

# Show plot
plt.show()

# Box plot with subgroups
# In this exercise, we'll look at the dataset containing responses from a survey given to young people.
# One of the questions asked of the young people was: "Are you interested in having pets?" Let's explore whether the
# distribution of ages of those answering "yes" is different from those answering "no", while taking into account
# whether the young person was getting support from their family.
survey_data = pd.read_csv('../../../data/6. eda_data_sources/young-people-survey-responses-pets.csv')

# Set palette to "Blues".
sns.set_palette("Blues")

# Adjust to add subgroups based on "Interested in Pets"
g = sns.catplot(x="Gender", y="Age", data=survey_data, kind="box", hue="Interested in Pets")

# Set the title of the plot to "Age of Those Interested in Pets vs. Not"
g.fig.suptitle("Age of Those Interested in Pets vs. Not", y=1.03)

# Show plot
plt.show()

survey_data = pd.read_csv('../../../data/6. eda_data_sources/young-people-survey-responses-techno.csv')

# Set the figure style to "dark".
sns.set_style("dark")

# Adjust to add subplots per gender
g = sns.catplot(x="Village - town", y="Likes Techno", data=survey_data, kind="bar", col="Gender")

# Set the title of the plot to "Percentage of Young People Who Like Techno"
g.fig.suptitle("Percentage of Young People Who Like Techno", y=1.03)

# Label x-axis as "Location of Residence" and y-axis as "% Who Like Techno"
g.set(xlabel="Location of Residence", ylabel="% Who Like Techno")

# Show plot
plt.show()