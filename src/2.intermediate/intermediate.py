import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# print() the last item from both the year and the pop list to see what the predicted population for the year 2100
# is. Use two print() functions. Before you can start, you should import matplotlib.pyplot as plt. pyplot is a
# sub-package of matplotlib, hence the dot. Use plt.plot() to build a line plot. year should be mapped on the
# horizontal axis, pop on the vertical axis. Don't forget to finish off with the plt.show() function to actually
# display the plot.

wbank = pd.read_csv("../../data/learning_python_sources/gapminder.csv")

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
lexp = pd.read_csv("../../data/learning_python_sources/life_expectancy.csv")
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

# Use tick_val and tick_lab as inputs to the xticks() function to make the plot more readable.
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
my_dict = {
    'Asia': 'red',
    'Europe': 'green',
    'Africa': 'blue',
    'Americas': 'yellow',
    'Oceania': 'black'
}

wbank['color'] = wbank['cont'].map(my_dict)
my_color = wbank['color'].tolist()

# Specify c and alpha inside plt.scatter() to color each bubble by continent and to
# make the bubbles 20% transparent. To see the different continents, also use the alpha
# argument inside plt.legend() to add transparency. This will make the legend of the plot
# more readable.

# Specify c and alpha inside plt.scatter()
plt.scatter(x=gdp_cap, y=life_exp, s=np.array(pop) / 570000, c=my_color, alpha=0.8)

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

# using dictionaries

# Use the index() method on countries to find the index of 'germany'. Store this index as ind_ger.
# Use ind_ger to access the capital of Germany from the capitals list. Print it out.

# Definition of countries and capital

countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# Get index of 'germany': ind_ger
ind_ger = countries.index('germany')

# Use ind_ger to print out capital of Germany
print(capitals[ind_ger])

# With the strings in countries and capitals, create a dictionary called europe with 4 key:value pairs. Beware of
# capitalization! Make sure you use lowercase characters everywhere.

# From string in countries and capitals, create dictionary europe
europe = {
    'spain': 'madrid',
    'france': 'paris',
    'germany': 'bonn',
    'norway': 'oslo'
}

# Print europe
print(europe)

# Check out which keys are in europe by calling the keys() method on europe. Print out the result.
# Print out the value that belongs to the key 'norway'.

# Print out the keys in europe
print(europe.keys())
print(europe.values())

# Print out value that belongs to key 'norway'
print(europe['norway'])

# Add the key 'italy' with the value 'rome' to europe.
# To assert that 'italy' is now a key in europe, print out 'italy' in europe.
# Add another key:value pair to europe: 'poland' is the key, 'warsaw' is the corresponding value.
# Print out europe.

# Add italy to europe
europe['italy'] = 'rome'

# Print out italy in europe
print('italy' in europe)

# Add poland to europe
europe['poland'] = 'warsaw'

# Print europe
print(europe)

# Update the capital of germany to 'berlin'.
# Remove australia from europe.
# Print out europe to see if your update worked.
europe = {'spain': 'madrid', 'france': 'paris', 'germany': 'bonn',
          'norway': 'oslo', 'italy': 'rome', 'poland': 'warsaw',
          'australia': 'vienna'}

print("------------------")
print(europe)
print("------------------")

# Update capital of germany
europe['germany'] = 'berlin'

# Remove australia
del (europe['australia'])

# Print europe
print(europe)

# Dictionary of dictionaries
europe = {'spain': {'capital': 'madrid', 'population': 46.77},
          'france': {'capital': 'paris', 'population': 66.03},
          'germany': {'capital': 'berlin', 'population': 80.62},
          'norway': {'capital': 'oslo', 'population': 5.084}}

# Use chained square brackets to select and print out the capital of France.
# Create a dictionary, named data, with the keys 'capital' and 'population'. Set them to 'rome' and 59.83, respectively.
# Add a new key-value pair to europe; the key is 'italy' and the value is data, the dictionary you just built.

# Print out the capital of France
print(europe['france']['capital'])

# Create sub-dictionary data
data = {'capital': 'rome', 'population': 59.83}

# Add data to europe under key 'italy'
europe['italy'] = data

# Print europe
print(europe)

# Pandas

# Use the pre-defined lists to create a dictionary called my_dict. There should be three key value pairs:
# key 'country' and value names.
# key 'drives_right' and value dr.
# key 'cars_per_cap' and value cpc.
# Use pd.DataFrame() to turn your dict into a DataFrame called cars.
# Print out cars and see how beautiful it is.

# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr = [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {
    'country': names,
    'drives_right': dr,
    'cars_per_cap': cpc
}

cars = pd.DataFrame(my_dict)

# Print cars
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

print(cars)

# Run the code with Run Code and assert that the first column should actually be used as row labels.
# Specify the index_col argument inside pd.read_csv(): set it to 0, so that the first column is used as row labels.
# Has the printout of cars improved now?

cars = pd.read_csv("../../data/learning_python_sources/cars.csv", index_col=0)
print(cars)
# Use single square brackets to print out the country column of cars as a Pandas Series. Use double square brackets
# to print out the country column of cars as a Pandas DataFrame. Use double square brackets to print out a DataFrame
# with both the country and drives_right columns of cars, in this order.

# Print out country column as Pandas Series
print(cars['country'])

# Print out country column as Pandas DataFrame
print(cars[['country']])

# Print out DataFrame with country and drives_right columns

print(type(cars[['country', 'drives_right']]))

# Print out first 3 observations
print(cars[0:3])

# Print out fourth, fifth and sixth observation
print(cars[3:6])

# using loc and iloc

# Use loc or iloc to select the observation corresponding to Japan as a Series. The label of this row is JPN,
# the index is 2. Make sure to print the resulting Series. Use loc or iloc to select the observations for Australia
# and Egypt as a DataFrame. You can find out about the labels/indexes of these rows by inspecting cars in the IPython
# Shell. Make sure to print the resulting DataFrame.

# Print out observation for Japan
print(cars.loc['JPN'])
print(cars.iloc[2])

# Print out observations for Australia and Egypt
print(cars.loc[['AUS', 'EG']])
print(cars.iloc[[1, 6]])

# Print out the drives_right value of the row corresponding to Morocco (its row label is MOR) Print out a
# sub-DataFrame, containing the observations for Russia and Morocco and the columns country and drives_right.

# Print out drives_right value of Morocco
print(cars.loc['MOR', 'drives_right'])
print(cars.iloc[5, 2])

# Print sub-DataFrame
print(cars.loc[['RU', 'MOR'], ['country', 'drives_right']])
print(cars.iloc[[4, 5], [0, 2]])

# Print out the drives_right column as a Series using loc or iloc.
# Print out the drives_right column as a DataFrame using loc or iloc.
# Print out both the cars_per_cap and drives_right column as a DataFrame using loc or iloc.

# Print out drives_right column as Series
print(cars.loc[:, 'drives_right'])
print(cars.iloc[:, 2])

# Print out drives_right column as DataFrame
print(cars.loc[:, ['drives_right']])
print(cars.iloc[:, [2]])

# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:, ['cars_per_cap', 'drives_right']])
print(cars.iloc[:, [1, 2]])
