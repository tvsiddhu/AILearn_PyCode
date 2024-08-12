# Import numpy as np
import numpy as np

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Set the seed
np.random.seed(123)

# Generate and print random numbers
print(np.random.rand())

# Use randint() to simulate a dice
print(np.random.randint(1, 7))

# Use randint() again
print(np.random.randint(1, 7))

# Roll the dice. Use randint() to create the variable dice.
# Finish the if-elif-else construct by replacing ___:
# If dice is 1 or 2, you go one step down.
# if dice is 3, 4 or 5, you go one step up.
# Else, you throw the dice again. The number on the dice is the number of steps you go up.
# Print out dice and step. Given the value of dice, was step updated correctly?

# Numpy is imported, seed is set
np.random.seed(123)

# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1, 7)

# Finish the control construct
if dice <= 2:
    step = step - 1
elif dice <= 5:
    step = step + 1
else:
    step = step + np.random.randint(1, 7)

# Print out dice and step
print(dice)
print(step)

# Random Walk
# Make a list random_walk that contains the first step, which is the integer 0.
# Finish the for loop:
# The loop should run 100 times.
# On each iteration, set step equal to the last element in the random_walk list. You can use the index -1 for this.
# Next, let the if-elif-else construct update step for you.
# The code that appends step to random_walk is already coded.
# Print out random_walk.

# Numpy is imported, seed is set

# Initialize random_walk
random_walk = [0]

# Complete the loop
for x in range(100):
    # Set step: last element in random_walk
    step = random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1, 7)

    # Determine next step
    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1, 7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)

# Use max() in a similar way to make sure that step doesn't go below zero if dice <= 2.
# Hit Submit Answer and check the contents of random_walk.

# Numpy is imported, seed is set

# Initialize random_walk
random_walk = [0]

# Complete the loop
for x in range(100):
    step = random_walk[-1]
    dice = np.random.randint(1, 7)

    if dice <= 2:
        # Replace below: use max to make sure step can't go below 0
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1, 7)

    random_walk.append(step)

print(random_walk)

# Visualize the walk
# Import matplotlib.pyplot as plt.
# Use plt.plot() to plot random_walk.
# Finish off with plt.show() to actually display the plot.

# Numpy is imported, seed is set

# Initialization
random_walk = [0]

for x in range(100):
    step = random_walk[-1]
    dice = np.random.randint(1, 7)

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1, 7)

    random_walk.append(step)

# Plot random_walk
plt.plot(random_walk)

# Show the plot
plt.show()

# Simulate multiple walks
# A single random walk is one thing, but that doesn't tell you if you have a good chance at winning the bet.
#
# To get an idea about how big your chances are of reaching 60 steps, you can repeatedly simulate the random walk and
# collect the results. That's exactly what you'll do in this exercise.
#
# The sample code already sets you off in the right direction. Another for loop is wrapped around the code you
# already wrote. It's up to you to add some bits and pieces to make sure all of the results are recorded correctly.
#
# Note: Don't change anything about the initialization of all_walks that is given. Setting any number inside the list
# will cause the exercise to crash!

# Numpy is imported, seed is set

# Initialize all_walks
all_walks = []

# Simulate random walk five times
for i in range(5):

    # Code from before
    random_walk = [0]
    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)

        random_walk.append(step)

    # Append random_walk to all_walks
    all_walks.append(random_walk)

# Print all_walks
print(all_walks)

# all_walks is a list of lists: every sub-list represents a single random walk. If you convert this list of lists to
# a NumPy array, you can start making interesting plots! matplotlib.pyplot is already imported as plt.
#
# The nested for loop is already coded for you - don't worry about it. For now, focus on the code that comes after
# this for loop.

# Use np.array() to convert all_walks to a NumPy array, np_aw. Try to use plt.plot() on np_aw. Also include plt.show().
# Does it work out of the box? Transpose np_aw by calling np.transpose() on np_aw. Call the result np_aw_t. Now
# every row in np_all_walks represents the position after 1 throw for the five random walks. Use plt.plot() to plot
# np_aw_t; also include a plt.show(). Does it look better this time?

# numpy and matplotlib imported, seed set.

# initialize and populate all_walks
all_walks = []
for i in range(10) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Convert all_walks to Numpy array: np_aw
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()

# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()

# Implement clumsiness
# With this neatly written code of yours, changing the number of times the random walk should be simulated is super easy.
# You simply update the range() function in the top-level for loop.

# There's still something we forgot! You're a bit clumsy and you have a 0.1% chance of falling down. That calls for
# another random number generation. Basically, you can generate a random float between 0 and 1. If this value is less
# than or equal to 0.001, you should reset step to 0.

# Change the range() function so that the simulation is performed 20 times.
# Finish the if condition so that step is set to 0 if a random float is less or equal to 0.005. Use np.random.rand().
# Make sure to include the clumsiness check.

# numpy and matplotlib imported, seed set

# clear the plot so it doesn't get cluttered if you run this many times
plt.clf()

# Simulate random walk 20 times
all_walks = []

for i in range(20) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)

        # Implement clumsiness
        if np.random.rand() <= 0.005:
            step = 0

        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()

# Plot the distribution All these fancy visualizations have put us on a sidetrack. We still have to solve the
# million-dollar problem: What are the odds that you'll reach 60 steps high on the Empire State Building?
# Basically, you want to know about the end points of all the random walks you've simulated. These end points have a
# certain distribution that you can visualize with a histogram.
# Note that if your code is taking too long to run, you might be plotting a histogram of the wrong data!

# To make sure we've got enough simulations, go crazy. Simulate the random walk 500 times.
# From np_aw_t, select the last row. This contains the endpoint of all 500 random walks you've simulated. Store this
# Numpy array as ends.
# Use plt.hist() to build a histogram of ends. Don't forget plt.show() to display the plot.

# numpy and matplotlib imported, seed set

# Simulate random walk 500 times
all_walks = []
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

# Select last row from np_aw_t: ends
ends = np_aw_t[-1]

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()

# Calculate the odds The histogram of the previous exercise was created from a NumPy array ends, that contains 500
# integers. Each integer represents the end point of a random walk. To calculate the chance that this end point is
# greater than or equal to 60, you can count the number of integers in ends that are greater than or equal to 60 and
# divide that number by 500, the total number of simulations.
#
# Well then, what's the estimated chance that you'll reach at least 60 steps high if you play this Empire State
# Building game? The ends array is everything you need; it's available in your Python session so you can make
# calculations in the IPython Shell.

# Calculate the estimated chance
# The ends array is available in your Python session.
# Calculate the fraction of simulations that ended in a result of 60 or higher.
# Print the result.

# numpy and matplotlib imported, seed set

# Simulate random walk 500 times
all_walks = []
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

# Select last row from np_aw_t: ends
ends = np_aw_t[-1]

# Calculate the fraction of simulations that ended in a result of 60 or higher
result = len(ends[ends >= 60]) / 500
print(result)
