import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 10. Interpreting line plots
# In this exercise, we'll continue to explore Seaborn's mpg dataset, which contains one row per car model and includes
# information such as the year the car was made, the number of miles per gallon ("M.P.G.") it achieves, the power of
# its engine (measured in "horsepower"), and its country of origin.
#
# How has the average miles per gallon achieved by these cars changed over time? Let's use line plots to find out!
mpg = pd.read_csv('../../data/eda_data_sources/mpg.csv')


# Create line plot
sns.relplot(x="model_year", y="mpg", data=mpg, kind="line")

# Show plot
plt.show()

# Which of the following is a correct interpretation of the line plot?
# Answer: The average miles per gallon has increased over time.

# 11. Visualizing standard deviation with line plots
# In the last exercise, we looked at how the average miles per
# gallon achieved by cars has changed over time. Now let's use a line plot to visualize how the distribution of miles
# per gallon has changed over time.

# Make the shaded area show the standard deviation
sns.relplot(x="model_year", y="mpg", data=mpg, kind="line", errorbar="sd")

# Show plot
plt.show()

# 12. Plotting subgroups in line plots
# Let's continue to look at the mpg dataset. We've seen that the average miles per gallon for cars has increased over
# time, but how has the average horsepower for cars changed over time? And does this trend differ by country of
# origin?

# Use relplot() and the mpg DataFrame to create a line plot with "model_year" on the x-axis and "horsepower" on the
# y-axis. Turn off the confidence intervals on the plot.
sns.relplot(x="model_year", y="horsepower", data=mpg, kind="line", errorbar=None)

# Show plot
plt.show()

# Create different lines for each country of origin ("origin") that vary in both line style and color.
sns.relplot(x="model_year", y="horsepower", data=mpg, kind="line", errorbar=None, style='origin', hue='origin')

# Show plot
plt.show()

# Add markers for each data point to the lines.
# Use the dashes parameter to use solid lines for all countries, while still allowing for different marker styles for
# each line.
sns.relplot(x="model_year", y="horsepower", data=mpg, kind="line", errorbar=None, style='origin', hue='origin',
            markers=True, dashes=False)

# Show plot
plt.show()

# 13. Count plots
# In this exercise, we'll return to exploring our dataset that contains the responses to a survey
# sent out to young people. We might suspect that young people spend a lot of time on the internet, but how much do
# they report using the internet each day? Let's use a count plot to break down the number of survey responses in
# each category and then explore whether it changes based on age.
#
# As a reminder, to create a count plot, we'll use the catplot() function and specify the name of the categorical
# variable to count (x=____), the pandas DataFrame to use (data=____), and the type of plot (kind="count").

# Import the data
survey_data = pd.read_csv('../../data/eda_data_sources/young-people-survey-responses.csv')

# Create column plot based on internet usage
sns.catplot(x="Internet usage", data=survey_data, kind="count")

# Show plot
plt.show()

# Make the bars horizontal instead of vertical.
sns.catplot(y="Internet usage", data=survey_data, kind="count")

# Show plot
plt.show()

# Separate this plot into two side-by-side column subplots based on "Age Category", which separates respondents into
# those that are younger than 21 vs. 21 and older.
sns.catplot(y="Internet usage", data=survey_data, kind="count", col='Age Category')

# Show plot
plt.show()

