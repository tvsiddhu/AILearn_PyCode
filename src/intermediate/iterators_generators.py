# Create a list of strings: flash
import pandas as pd

flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']
# Print each list item in flash using a for loop
for person in flash:
    print(person)

# Create an iterator for flash: superhero
print("------")
superhero = iter(flash)

# Print each item from the iterator
print(next(superhero))
print(next(superhero))
print(next(superhero))
print(next(superhero))

# You can use range() in a for loop as if it's a list to be iterated over:
#
# for i in range(5): print(i) Recall that range() doesn't actually create the list; instead, it creates a range
# object with an iterator that produces the values until it reaches the limit (in the example, until the value 4). If
# range() created the actual list, calling it with a value of 10^100 may not work, especially since a number as big
# as that may go over a regular computer's memory. The value 10^100 is actually what's called a Googol which is a 1
# followed by a hundred 0s. That's a huge number!
#
# Your task for this exercise is to show that calling range() with 10^100 won't actually pre-create the list.

# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
print("------")
for num in range(3):
    print(num)

# Create an iterator for range(10 ** 100): googol
googol = iter(range(10 ** 100))

# Print the first 5 values from googol
print("------")
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))

# Iterators as function arguments You've been using the iter() function to get an iterator object, as well as the
# next() function to retrieve the values one by one from the iterator object.
#
# There are also functions that take iterators and iterables as arguments. For example, the list() and sum()
# functions return a list and the sum of elements, respectively.
#
# In this exercise, you will use these functions by passing an iterable from range() and then printing the results of
# the function calls.

# Create a range object: values
values = range(10, 21)

# Print the range object
print("------")
print(values)

# Create a list of integers: values_list
values_list = list(values)

# Print values_list
print(values_list)

# Get the sum of values: values_sum
values_sum = sum(values)

# Print values_sum
print(values_sum)

# Using enumerate() You're really getting the hang of using iterators, great job!
#
# You've just gained several new ideas on iterators from the last video and one of them is the enumerate() function.
# Recall that enumerate() returns an enumerate object that produces a sequence of tuples, and each of the tuples is an
# index-value pair.
#
# In this exercise, you are given a list of strings mutants and you will practice using enumerate() on it by printing
# out a list of tuples and unpacking the tuples using a for loop.

# Create a list of strings: mutants
mutants = ['charles xavier', 'bobby drake', 'kurt wagner', 'max eisenhardt', 'kitty pryde']

# Create a list of tuples: mutant_list
print("------")
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
print("------")
for index1, value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
print("------")
for index2, value2 in enumerate(mutants, start=1):
    print(index2, value2)

# Using zip() You're probably wondering when you can use all these functions. Here's one example where you can use
# zip() and enumerate() to achieve a common goal by producing a list of tuples.
#
# Let's say you have a list of strings representing mutants and a list of strings representing powers. You want to
# pair each mutant with their corresponding power, which are stored in the lists mutants and powers. Here, you'll use
# the zip() function to accomplish this.

# Create a list of tuples: mutant_data
powers = ['telepathy', 'thermokinesis', 'teleportation', 'magnetokinesis', 'intangibility']
aliases = ['prof x', 'iceman', 'nightcrawler', 'magneto', 'shadowcat']
mutant_data = list(zip(mutants, aliases, powers))

# Print the list of tuples
print("------")
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)

# Print the zip object
print(mutant_zip)

# Unpack the zip object and print the tuple values
print("------")
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)

# Using * and zip to 'unzip' You know how to use zip() as well as how to print out values from a zip object. Excellent!
#
# Let's play around with zip() a little more. There is no unzip function for doing the reverse of what zip() does.
# We can, however, reverse what has been zipped together by using zip() with a little help from *! * unpacks an
# iterable such as a list or a tuple into positional arguments in a function call.

# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# Print the zip object
print("------")
print(z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)

# Check if unpacked tuples are equivalent to original tuples
print("------")
print(result1 == mutants)
print(result2 == powers)

# Sometimes, the data we have to process reaches a size that is too much for a computer's memory to handle. This is a
# common problem faced by data scientists. A solution to this is to process an entire data source chunk by chunk,
# instead of a single go all at once.
#
# In this exercise, you will do just that. You will process a large csv file of Twitter data in the same way that you
# processed 'tweets.csv' in Bringing it all together exercises of the prequel course, but this time, working on it in
# chunks of 10 entries at a time.

# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('../../data/learning_python_sources/tweets.csv', chunksize=10):
    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print("------")
print(counts_dict)

# Define count_entries()
def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of occurrences as value for each key."""
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file, chunksize=c_size):
        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict

# Call count_entries(): result_counts
result_counts = count_entries('../../data/learning_python_sources/tweets.csv', 10, 'lang')

# Print result_counts
print("------")
print(result_counts)

