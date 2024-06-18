import numpy as np
import pandas as pd

# Equality Exercises

# Comparison of booleans
print(True == False)

# Comparison of integers
print(-5 * 15 != 75)

# Comparison of strings
print('pyscript' == 'PyScript')

# Compare a boolean with an integer
print(True == 1)

# Comparison of integers
x = -3 * 6
print(x >= -10)

# Comparison of strings
y = "test"
print("test" <= y)

# Comparison of booleans
print(True > False)

# Create arrays
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18
print(my_house >= 18)

# my_house less than your_house
print(my_house < your_house)

# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?
print(10 < my_kitchen < 18)

# my_kitchen smaller than 14 or bigger than 17?
print(my_kitchen < 14 or my_kitchen > 17)

# Double my_kitchen smaller than triple your_kitchen?
print(my_kitchen * 2 < your_kitchen * 3)

print("---------------------------------")

# and, or, not
x = 8
y = 9
print(not (not (x < 3) and not (y > 14 or y > 10)))

print("using np.logical_or, np.logical_and")
# Create arrays
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house > 18.5, my_house < 10))

# Both my_house and your_house smaller than 11
print(np.logical_and(my_house < 11, your_house < 11))

print("---------if else elif---------")
# if else elif

area = 10.0
if area < 9:
    print("small")
elif area < 12:
    print("medium")
else:
    print("large")

# Examine the if statement that prints out "looking around in the kitchen." if room equals "kit".
# Write another if statement that prints out "big place!" if area is greater than 15.

room = "kit"
area = 14

if room == "kit":
    print("looking around in the kitchen.")
if area > 15:
    print("big place!")
else:
    print("pretty small.")

print("---------filtering pandas dataframes---------")
# Extract the drives_right column as a Pandas Series and store it as dr.
# Use dr, a boolean Series, to subset the cars DataFrame. Store the resulting selection in sel.
# Print sel, and assert that drives_right is True for all observations.

cars = pd.read_csv('../data/cars.csv', index_col=0)
dr = cars['drives_right']
sel = cars[dr]
print(sel)
assert sel['drives_right'].all() == True

# convert the code to a one-liner
sel = cars[cars['drives_right']]
print(sel)
assert sel['drives_right'].all() == True

# Select the cars_per_cap column from cars as a Pandas Series and store it as cpc. Use cpc in combination with a
# comparison operator and 500. You want to end up with a boolean Series that's True if the corresponding country has
# a cars_per_cap of more than 500 and False otherwise. Store this boolean Series as many_cars. Use many_cars to
# subset cars, similar to what you did before. Store the result as car_maniac. Print out car_maniac to see if you got
# it right.

cpc = cars['cars_per_cap']
many_cars = cpc > 500
car_maniac = cars[many_cars]
print(car_maniac)

# Use the code sample provided to create a DataFrame medium, that includes all the observations of cars that have a
# cars_per_cap between 100 and 500. Print out medium.

medium = cars[np.logical_and(cars['cars_per_cap'] > 100, cars['cars_per_cap'] < 500)]
print(medium)
