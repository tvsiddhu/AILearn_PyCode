import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import student data
student_data = pd.read_csv('../../../data/6. eda_data_sources/student-alcohol-consumption.csv')

# 16. Create and interpret a box plot Let's continue using the student_data dataset. In an earlier exercise,
# we explored the relationship between studying and final grade by using a bar plot to compare the average final
# grade ("G3") among students in different categories of "study_time".
#
# In this exercise, we'll try using a box plot look at this relationship instead. As a reminder, to create a box plot
# you'll need to use the catplot() function and specify the name of the categorical variable to put on the x-axis (
# x=____), the name of the quantitative variable to summarize on the y-axis (y=____), the pandas DataFrame to use (
# data=____), and the type of plot (kind="box").

# Specify the category ordering in the "study_time" column
study_time_order = ["<2 hours", "2 to 5 hours", "5 to 10 hours", ">10 hours"]

# Create a box plot with "study_time" on the x-axis and "G3" on the y-axis
sns.catplot(x="study_time", y="G3", data=student_data, kind="box", order=study_time_order)

# Show plot
plt.show()

# Which of the following is a correct interpretation of the box plot?
# Answer: The median final grade ("G3") is highest for students who study 2 to 5 hours.

# 17. Omitting outliers Now let's use the student_data dataset to compare the distribution of final grades ("G3")
# between students who have internet access at home and those who don't. To do this, we'll use the "internet"
# variable, which is a binary (yes/no) indicator of whether the student has internet access at home.
#
# Since internet may be less accessible in rural areas, we'll add subgroups based on where the student lives. For
# this, we can use the "location" variable, which is an indicator of whether a student lives in an urban ("Urban") or
# rural ("Rural") location.

# Create a box plot with subgroups and omit the outliers
sns.catplot(x="internet", y="G3", data=student_data, kind="box", hue="location")

# Show plot
plt.show()

# 18. Adjusting the whiskers In the lesson we saw that there are multiple ways to define the whiskers in a box plot.
# In this set of exercises, we'll continue to use the student_data dataset to compare the distribution of final
# grades ("G3") between students who are in a romantic relationship and those that are not. We'll use the "romantic"
# variable, which is a yes/no indicator of whether the student is in a romantic relationship.
#
# Let's create a box plot to look at this relationship and try different ways to define the whiskers.

# Adjust the code to make the box plot whiskers to extend to 0.5 * IQR. Recall: the IQR is the interquartile range.
sns.catplot(x="romantic", y="G3", data=student_data, kind="box", whis=0.5)

# Show plot
plt.show()

# Change the code to set the whiskers to extend to the 5th and 95th percentiles.
sns.catplot(x="romantic", y="G3", data=student_data, kind="box", whis=[5, 95])

# Show plot
plt.show()

# Change the code to set the whiskers to extend to the min and max values.
sns.catplot(x="romantic", y="G3", data=student_data, kind="box", whis=[0, 100])

# Show plot
plt.show()
