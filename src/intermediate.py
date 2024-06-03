
import pandas as pd
import matplotlib.pyplot as plt

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

