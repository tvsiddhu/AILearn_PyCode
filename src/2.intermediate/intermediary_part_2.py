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

cars = pd.read_csv('../../data/1.learning_python_sources/cars.csv', index_col=0)
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

# Loops

# Initialize offset
offset = 8
while offset != 0:
    print("correcting...")
    offset = offset - 1
    print(offset)

# Initialize offset
offset = -6

# Code the while loop
while offset != 0:
    print("correcting...")
    if offset > 0:
        offset = offset - 1
    else:
        offset = offset + 1
    print(offset)

    # For Loops

    # areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for area in areas:
    print(area)

# Indexes and values
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate()
for index, area in enumerate(areas):
    print("room " + str(index+1) + ": " + str(area))

# Loop over list of lists
# house list of lists
house = [["hallway", 11.25],
         ["kitchen", 18.0],
         ["living room", 20.0],
         ["bedroom", 10.75],
         ["bathroom", 9.50]]

# Build a for loop from scratch
for room in house:
    print("the " + room[0] + " is " + str(room[1]) + " sqm")

# Loop over dictionary
# Definition of dictionary
europe = {'spain': 'madrid', 'france': 'paris', 'germany': 'berlin',
          'norway': 'oslo', 'italy': 'rome', 'poland': 'warsaw', 'austria': 'vienna'}

# Iterate over europe
for key, value in europe.items():
    print("the capital of " + key + " is " + value)

# Loop over Numpy array
# Import numpy as np

baseball = pd.read_csv("../../data/1.learning_python_sources/baseball.csv")[['Height', 'Weight']].to_numpy().tolist()
np_baseball = np.array(baseball)

np_height = np_baseball[:, 0]

# For loop over np_height
for height in np_height:
    print(str(height) + " inches")

# For loop over np_baseball
for baseball in np.nditer(np_baseball):
    print(baseball)

# Import cars data
cars = pd.read_csv('../../data/1.learning_python_sources/cars.csv', index_col=0)

# Iterate over rows of cars
for lab, row in cars.iterrows():
    print(lab)
    print(row)

# Adapt for loop
for lab, row in cars.iterrows():
    print(lab + ": " + str(row['cars_per_cap']))

# Use a for loop to add a new column, named COUNTRY, that contains a uppercase version of the country names in the
# "country" column. You can use the string method upper() for this.

# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows():
    cars.loc[lab, "COUNTRY"] = row['country'].upper()

# Replace the for loop with a one-liner that uses .apply(str.upper). The call should give the same result: a column
# COUNTRY should be added to cars, containing an uppercase version of the country names.
# Use the .apply(str.upper) method on cars["country"] and store the result in the column "COUNTRY".

cars["COUNTRY"] = cars["country"].apply(str.upper)
print(cars)