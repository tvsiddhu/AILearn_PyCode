# Import the math package
import math
from math import radians

import numpy as np
import pandas as pd

monthly_savings = 10
num_months = 12
intro = "Hello! How are you?"

# Calculate year_savings using monthly_savings and num_months
year_savings = monthly_savings * num_months
print(year_savings)

# Print the type of year_savings
print(type(year_savings))

# Assign sum of intro and intro to doubleintro
doubleintro = intro + intro

# Print out doubleintro
print(doubleintro)

a_list = [1, 3, 4, 2]
b_list = [[1, 2, 3], [4, 5, 7]]
c_list = [1 + 2, "a" * 5, 3]

print(a_list)
print(b_list)
print(c_list)

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom", bed],
         ["bathroom", bath]]

# Print out house
print(house)

# Print out the type of house
print(type(house))

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Sum of kitchen and bedroom area: eat_sleep_area
eat_sleep_area = areas[3] + areas[7]

# Print the variable eat_sleep_area
print(eat_sleep_area)

# Use slicing to create downstairs
downstairs = areas[:6]

# Use slicing to create upstairs
upstairs = areas[-4:]

# Print out downstairs and upstairs
print(downstairs)
print(upstairs)

house = [['hallway', 11.25], ['kitchen', 18.0], ['living room', 20.0], ['bedroom', 10.75], ['bathroom', 9.5]]

# The answer should be 9.5 for the index [-1][-1]
print(house[-1][-1])

# Change the values in the areas list
areas[-1] = 10.50
areas[4] = "chill zone"

# Add poolhouse data to areas, new list is areas_1 Use the + operator to paste the list ["poolhouse", 24.5] to the
# end of the areas list. Store the resulting list as areas_1. Further, extend areas_1 by adding data on your garage.
# Add the string "garage" and float 15.45. Name the resulting list areas_2.

poolhouse = ["poolhouse", 24.5]
areas_1 = areas + poolhouse

areas_2 = areas_1 + ["garage", 15.45]

areas = areas_2
print(areas)

# Remove the corresponding string and float from the areas list. Use the del statement to remove the poolhouse data.

del (areas[-4:-2])
print(areas)

# Change the second command, that creates the variable areas_copy, such that areas_copy is an explicit copy of areas.
# After your edit, changes made to areas_copy shouldn't affect areas. Submit the answer to check this.

areas = [11.25, 18.0, 20.0, 10.75, 9.50]
areas_copy = list(areas)
areas_copy[0] = 5.0
print(areas)
print(areas_copy)

# Use print() in combination with type() to print out the type of var1.
# Use len() to get the length of the list var1. Wrap it in a print() call to directly print it out.
# Use int() to convert var2 to an integer. Store the output as out2.

var1 = [1, 2, 3, 4]
var2 = True
print(type(var1))
print(len(var1))
out2 = int(var2)

print(type(out2))

# Use + to merge the contents of first and second into a new list: full.
# Call sorted() on full and specify the reverse argument to be True. Save the sorted list as full_sorted.
# Finish off by printing out full_sorted.

first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]
full = first + second
full_sorted = sorted(full, reverse=True)
print(full_sorted)

# Use the upper() method on place and store the result in place_up. Use the syntax for calling methods that you
# learned in the previous video. Print out place and place_up. Did both change? Print out the number of o's on the
# variable place by calling count() on place and passing the letter 'o' as an input to the method. We're talking
# about the variable place, not the word "place"!

place = "poolhouse"
place_up = place.upper()
print(place)
print(place_up)
print(place.count('o'))

# Use the index() method to get the index of the element in areas that is equal to 20.0. Print out this index.
# Call count() on areas to find out how many times 9.50 appears in the list. Again, simply print out this number.

areas = [11.25, 18.0, 20.0, 10.75, 9.50]
print(areas.index(20.0))
print(areas.count(9.50))

# Use append() twice to add the size of the poolhouse and the garage again: 24.5 and 15.45, respectively. Make sure
# to add them in this order. Print out areas Use the reverse() method to reverse the order of the elements in areas.
# Print out areas once more.

areas.append(24.5)
areas.append(15.45)
print(areas)
areas.reverse()
print(areas)

# Packages
# Import the math package. Now you can access the constant pi with math.pi.
# Calculate the circumference of the circle and store it in C.
# Calculate the area of the circle and store it in A.


# Definition of radius
r = 0.43

# Calculate C
C = 2 * math.pi * r

# Calculate A
A = math.pi * r ** 2
print("Circumference: " + str(C))
print("Area: " + str(A))

# Perform a selective import from the math package where you only import the radians function. Calculate the distance
# travelled by the Moon over 12 degrees of its orbit. Assign the result to dist. You can calculate this as r * phi,
# where r is the radius and phi is the angle in radians. To convert an angle in degrees to an angle in radians,
# use the radians() function, which you just imported. Print out dist.

# Import radians function of math package


r = 192500
phi = radians(12)
dist = r * phi
print(dist)

# Import the numpy package as np, so that you can refer to numpy with np.
# Use np.array() to create a numpy array from baseball. Name this array np_baseball.
# Print out the type of np_baseball to check that you got it right.

# Read baseball data from CSV file


baseball = [180, 215, 210, 210, 188, 176, 209, 200]

np_baseball = np.array(baseball)
print(type(np_baseball))

# Create a numpy array from height_in. Name this new array np_height_in. Print np_height_in. Multiply np_height_in
# with 0.0254 to convert all height measurements from inches to meters. Store the new values in a new array,
# np_height_m. Print out np_height_m and check if the output makes sense.

# Create a numpy array from height: np_height_in

# Original height in inches data is available within the height_in list in numpy package. This is just a simplified
# version of the data.


mlb = pd.read_csv("../../data/learning_python_sources/baseball.csv")

height_in = mlb['Height'].tolist()
weight_lb = mlb['Weight'].tolist()

np_height_in = np.array(height_in)

# Print out np_height_in
print(np_height_in)

# Convert np_height_in to m: np_height_m
np_height_m = np_height_in * 0.0254

# Print np_height_m
print(np_height_m)

# Create a numpy array from the weight_lb list with the correct units. Multiply by 0.453592 to go from pounds to
# kilograms. Store the resulting numpy array as np_weight_kg. Use np_height_m and np_weight_kg to calculate the BMI
# of each player. Use the following equation: BMI=weight(kg) / height(m)2 Save the resulting numpy array as bmi.
# Print out bmi.

# Original weight in pounds data is available within the weight_lb list in numpy package. This is just a simplified
# version of the data

np_weight_lb = np.array(weight_lb)

# Print out np_weight_lb
print(np_weight_lb)

# Convert np_weight_lb to kg: np_weight_kg
np_weight_kg = np_weight_lb * 0.453592

# Print np_weight_kg
print(np_weight_kg)

# Calculate the BMI: bmi
bmi = np_weight_kg / (np_height_m ** 2)

# Print out bmi
print("BMI:")
print(bmi)

# Create a boolean numpy array: the element of the array should be True if the corresponding baseball player's BMI is
# below 21. You can use the < operator for this. Name the array light. Print the array light. Print out a numpy array
# with the BMIs of all baseball players whose BMI is below 21. Use light inside square brackets to do a selection on
# the bmi array.

# Create the light array
light = bmi < 21

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])

# Have a look at this line of code:
#
# np.array([True, 1, 2]) + np.array([3, 4, False]) Can you tell which code chunk builds the exact same Python object?
# The numpy package is already imported as np, so you can start experimenting in the IPython Shell straight away!
# np.array([4, 3, 0]) + np.array([0, 2, 2])
# np.array([1, 1, 2]) + np.array([3, 4, -1])
# np.array([0, 1, 2]) + np.array([3, 4, 5])
# np.array([1, 1, 2]) + np.array([3, 4, 0])

print("Expected Output:")
print(np.array([True, 1, 2]) + np.array([3, 4, False]))
print("Option 1:")
print(np.array([True, 1, 2, 3, 4, False]))
print("Option 2:")
print(np.array([4, 3, 0]) + np.array([0, 2, 2]))
print("Option 3:")
print(np.array([1, 1, 2]) + np.array([3, 4, -1]))
print("Option 4:")
print(np.array([0, 1, 2, 3, 4, 5]))

# Subsetting NumPy Arrays
# Subset np_weight_lb by printing out the element at index 50.
# Print out a sub-array of np_height_in that contains the elements at index 100 up to and including index 110.


np_weight_lb = np.array(weight_lb)

# Print out the weight at index 50
print(np_weight_lb[15])

np_height_in = np.array(height_in)

# Print out the weight at index 100 up to and including index 110
print(np_height_in[4:8])

# Use np.array() to create a 2D numpy array from baseball. Name it np_baseball.
# Print out the type of np_baseball.
# Print out the shape attribute of np_baseball. Use np_baseball.shape.

baseball = pd.read_csv("../../data/learning_python_sources/baseball.csv")[['Height', 'Weight']].to_numpy().tolist()

np_baseball = np.array(baseball)
print(type(np_baseball))
print(np_baseball.shape)

# Print out the 50th row of np_baseball.
# Make a new variable, np_weight_lb, containing the entire second column of np_baseball.
# Select the height (first column) of the 124th baseball player in np_baseball and print it out.

# Print out the 50th row of np_baseball
print(np_baseball[49, :])

# Select the entire second column of np_baseball: np_weight_lb
np_weight_lb = np_baseball[:, 1]

# Print out height of 124th player
print(np_baseball[123, 0])

# You managed to get hold of the changes in height, weight and age of all baseball players. It is available as a 2D
# numpy array, updated. Add np_baseball and updated and print out the result. You want to convert the units of height
# and weight to metric (meters and kilograms, respectively). As a first step, create a numpy array with three values:
# 0.0254, 0.453592 and 1. Name this array conversion. Multiply np_baseball with conversion and print out the result.

baseball = pd.read_csv("../../data/learning_python_sources/baseball.csv")[['Height', 'Weight', 'Age']].to_numpy().tolist()
n = len(baseball)
updated = np.array(pd.read_csv("../../data/learning_python_sources/update.csv", header=None))

np_baseball = np.array(baseball)

# Print out addition of np_baseball and updated
print(np_baseball + updated)

# Create numpy array: conversion
conversion = np.array([0.0254, 0.453592, 1])

# Print out product of np_baseball and conversion
print(np_baseball * conversion)

# Create numpy array np_height_in that is equal to first column of np_baseball.
# Print out the mean of np_height_in.
# Print out the median of np_height_in.

np_height_in = np_baseball[:, 0]

# Print out the mean of np_height_in
print("Mean Height:")
print(np.mean(np_height_in))

# Print out the median of np_height_in
print("Median Height:")
print(np.median(np_height_in))

# The code to print out the mean height is already included. Complete the code for the median height. Replace None
# with the correct code. Use np.std() on the first column of np_baseball to calculate stddev. Replace None with the
# correct code. Do big players tend to be heavier? Use np.corrcoef() to store the correlation between the first and
# second column of np_baseball in corr. Replace None with the correct code.

# Print mean height (first column)
avg = np.mean(np_baseball[:, 0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:, 0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:, 0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:, 0], np_baseball[:, 1])
print("Correlation: " + str(corr))

# Convert heights and positions, which are regular lists, to numpy arrays. Call them np_heights and np_positions.
# Extract all the heights of the goalkeepers. You can use a little trick here: use np_positions == 'GK' as an index
# for np_heights. Assign the result to gk_heights. Extract all the heights of all the other players. This time use
# np_positions != 'GK' as an index for np_heights. Assign the result to other_heights. Print out the median height of
# the goalkeepers using np.median(). Replace None with the correct code. Do the same for the other players. Print out
# their median height. Replace None with the correct code.

# convert heights and positions to numpy arrays: np_heights, np_positions
fifa = pd.read_csv("../../data/learning_python_sources/fifa.csv", skipinitialspace=True, usecols=['position', 'height'])
positions = list(fifa.position)
heights = list(fifa.height)

np_heights = np.array(heights)
np_positions = np.array(positions)

# Heights of the goalkeepers: gk_heights
gk_heights = np_heights[np_positions == 'GK']

# Heights of the other players: other_heights
other_heights = np_heights[np_positions != 'GK']

# Print out the median height of goalkeepers. Replace 'None'
print("Median height of goalkeepers: " + str(np.median(gk_heights)))

# Print out the median height of other players. Replace 'None'
print("Median height of other players: " + str(np.median(other_heights)))
