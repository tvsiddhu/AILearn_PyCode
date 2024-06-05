import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# print() the last item from both the year and the pop list to see what the predicted population for the year 2100
# is. Use two print() functions. Before you can start, you should import matplotlib.pyplot as plt. pyplot is a
# sub-package of matplotlib, hence the dot. Use plt.plot() to build a line plot. year should be mapped on the
# horizontal axis, pop on the vertical axis. Don't forget to finish off with the plt.show() function to actually
# display the plot.

wbank = pd.read_csv("../data/gapminder.csv")

year = wbank['country'].tolist()
pop = wbank['population'].tolist()

# Print the last item from year and pop
print(year[-1])
print(pop[-1])

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year, pop)

# Display the plot with plt.show()
plt.show()

# Print the last item from both the list gdp_cap, and the list life_exp; it is information about Zimbabwe. Build a
# line chart, with gdp_cap on the x-axis, and life_exp on the y-axis. Does it make sense to plot this data on a line
# plot? Don't forget to finish off with a plt.show() command, to actually display the plot.

gdp_cap = wbank['gdp_cap'].tolist()
life_exp = wbank['life_exp'].tolist()
cont = wbank['cont'].tolist()
# read the life expectancy data for the year 1950
lexp = pd.read_csv("../data/life_expectancy.csv")
life_exp1950 = lexp['1950'].tolist()

# Print the last item of gdp_cap and life_exp
print(gdp_cap[-1])
print(life_exp[-1])

# Make a line plot, gdp_cap on the x-axis, life_exp on the y-axis
plt.plot(gdp_cap, life_exp)

# Display the plot
plt.show()

# Change the line plot that's coded in the script to a scatter plot. A correlation will become clear when you display
# the GDP per capita on a logarithmic scale. Add the line plt.xscale('log'). Finish off your script with plt.show()
# to display the plot.

# Change the line plot below to a scatter plot
plt.scatter(gdp_cap, life_exp)

# Put the x-axis on a logarithmic scale
plt.xscale('log')

# Show plot
plt.show()

# Start from scratch: import matplotlib.pyplot as plt.
# Build a scatter plot, where pop is mapped on the horizontal axis, and life_exp is mapped on the vertical axis.
# Finish the script with plt.show() to actually display the plot. Do you see a correlation?

# Build Scatter plot
plt.scatter(pop, life_exp)

# Show plot
plt.show()

# Use plt.hist() to create a histogram of the values in life_exp. Do not specify the number of bins; Python will set
# the number of bins to 10 by default for you. Add plt.show() to actually display the histogram. Can you tell which
# bin contains the most observations?

# Create histogram of life_exp data
plt.hist(life_exp)

# Display histogram
plt.show()

# Build a histogram of life_exp, with 5 bins. Can you tell which bin contains the most observations?
# Build another histogram of life_exp, this time with 20 bins. Is this better?

# Build histogram with 5 bins
plt.hist(life_exp, bins=5)

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(life_exp, bins=20)

# Show and clean up again
plt.show()
plt.clf()

# Build a histogram of life_exp with 15 bins.
# Build a histogram of life_exp1950, also with 15 bins. Is there a big difference with the histogram for the 2007 data?

# Histogram of life_exp, 15 bins
plt.hist(life_exp, bins=15)

# Show and clear plot
plt.show()
plt.clf()

# Histogram of life_exp1950, 15 bins
plt.hist(life_exp1950, bins=15)

# Show and clear plot
plt.show()
plt.clf()

# The strings xlab and ylab are already set for you. Use these variables to set the label of the x- and y-axis.
# The string title is also coded for you. Use it to add a title to the plot.
# After these customizations, finish the script with plt.show() to actually display the plot.

# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log')

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

# Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)

# Add title
plt.title(title)

# After customizing, display the plot
plt.show()

# Use tick_val and tick_lab as inputs to the xticks() function to make the the plot more readable.
# As usual, display the plot with plt.show() after you've added the customizations.

# Scatter plot
plt.scatter(gdp_cap, life_exp)

# Previous customizations
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')

# Definition of tick_val and tick_lab
tick_val = [1000, 10000, 100000]
tick_lab = ['1k', '10k', '100k']

# Adapt the ticks on the x-axis
plt.xticks(tick_val, tick_lab)

# After customizing, display the plot
plt.show()

# Looks good, but increasing the size of the bubbles will make things stand out more. Import the numpy package as np.
# Use np.array() to create a numpy array from the list pop. Call this NumPy array np_pop. Double the values in np_pop
# setting the value of np_pop equal to np_pop * 2. Because np_pop is a NumPy array, each array element will be
# doubled. Change the s argument inside plt.scatter() to be np_pop instead of pop.

# Store pop as a numpy array: np_pop
np_pop = np.array(pop)

# # Double np_pop
# np_pop = np_pop * 2

# Adjust np_pop
np_pop = np_pop / 570000

# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s=np_pop)

# Previous customizations
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000], ['1k', '10k', '100k'])

# Display the plot
plt.show()

# map continents onto colors
dict = {
    'Asia':'red',
    'Europe':'green',
    'Africa':'blue',
    'Americas':'yellow',
    'Oceania':'black'
}

wbank['color'] = wbank['cont'].map(dict)
cont = wbank['color'].tolist()

# Specify c and alpha inside plt.scatter() to color each bubble by continent and to
# make the bubbles 20% transparent. To see the different continents, also use the alpha
# argument inside plt.legend() to add transparency. This will make the legend of the plot
# more readable.

# Specify c and alpha inside plt.scatter()
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) / 570000, c = cont, alpha = 0.8)

# Previous customizations
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000], ['1k', '10k', '100k'])
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')
plt.grid(True)

# Display the plot
plt.show()

