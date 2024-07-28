# Import pandas
import pandas as pd


def shout() -> object:
    """
    Print a string with three exclamation marks
    :return:
    """

    shout_word = "congratulations" + "!!!"
    print(shout_word)


# Call the function shout
shout()


# Define the function shout with the parameter, word
def shout(word):
    """
    Return a string with three exclamation marks
    :param word:
    :return:
    """
    shout_word = word + "!!!"
    return shout_word


# Call shout with the string argument 'congratulations'
yell = shout('congratulations')
assert isinstance(yell, object)
print(yell)


# You are now going to use what you've learned to modify the shout() function further. Here, you will modify shout()
# to accept two arguments. Parts of the function shout(), which you wrote earlier, are shown.


# Define shout with parameters word1 and word2
def shout(word1, word2):
    """
    Concatenate word1 with '!!!' and word2 with '!!!' and return the result
    :param word1:
    :param word2:
    :return:
    """
    shout1 = word1 + "!!!"
    shout2 = word2 + "!!!"
    new_shout = shout1 + shout2
    return new_shout


# Pass 'congratulations' and 'you' to shout(): yell
yell = shout('congratulations', 'you')
assert isinstance(yell, object)
print(yell)

# Unpack the tuple nums into num1, num2, and num3
nums = (3, 4, 6)
num1, num2, num3 = nums

# Construct even_nums
even_nums = (2, num2, num3)
assert isinstance(even_nums, tuple)
print(even_nums)


# Functions that return multiple values
# Define shout_all with parameters word1 and word2
def shout_all(word1, word2):
    """
    Return a tuple of strings concatenated with '!!!'
    :param word1:
    :param word2:
    :return:
    """
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + "!!!"
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + "!!!"
    # Construct a tuple with shout1 and shout2: shout_words
    shout_words = (shout1, shout2)
    # Return shout_words
    return shout_words


# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1, yell2 = shout_all('congratulations', 'you')
assert isinstance(yell1, object)
assert isinstance(yell2, object)
print(yell1)
print(yell2)

# Bringing it all together (1) For this exercise, your goal is to recall how to load a dataset into a DataFrame. The
# dataset contains Twitter data, and you will iterate over entries in a column to build a dictionary in which the keys
# are the names of languages and the values are the number of tweets in the given language. The file tweets.csv is
# available in your current directory.


# Import Twitter data as DataFrame: df
df = pd.read_csv('../data/tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:
    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry] += 1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)


# Bringing it all together (2) Great job! You've now defined the functionality for iterating over entries in a column
# and building a dictionary with keys the names of languages and values the number of tweets in the given language. In
# this exercise, you will define a function with the functionality you developed in the previous exercise, return the
# resulting dictionary from within the function, and call the function with the appropriate arguments.

# Define count_entries()
def count_entries(df, col_name):
    """
    Return a dictionary with counts of occurrences as value for each key
    :param df:
    :param col_name:
    :return:
    """
    # Initialize an empty dictionary: langs_count
    langs_count = {}
    # Extract column from DataFrame: col
    col = df[col_name]
    # Iterate over lang column in DataFrame
    for entry in col:
        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry] += 1
        # Else add the language to langs_count, set the value to 1
        else:
            langs_count[entry] = 1
    # Return the populated dictionary
    return langs_count


# Call count_entries(): result
result = count_entries(df, 'lang')

# Print the result
print(result)

# Create a string: team
team = "teen titans"


# The keyword global Let's work more on your mastery of scope. In this exercise, you will use the keyword global
# within a function to alter the value of a variable defined in the global scope

# Define change_team()
def change_team():
    """
    Change the value of the global variable team
    :return:
    """
    # Use team in global scope
    global team
    # Change the value of team in global: team
    team = "justice league"


# Print team
print(team)


# Nested Functions I
# Define three_shouts
def three_shouts(word1, word2, word3):
    """
    Return a tuple of strings concatenated with '!!!'
    :param word1:
    :param word2:
    :param word3:
    :return:
    """

    # Define inner
    def inner(word):
        """
        Return a string concatenated with '!!!'
        :param word:
        :return:
        """
        return word + '!!!'

    # Return a tuple of strings
    return inner(word1), inner(word2), inner(word3)


# Call three_shouts() and print
print(three_shouts('a', 'b', 'c'))


# Nested Functions II
# Define echo
def echo(n):
    """
    Return the inner_echo function
    :param n:
    :return:
    """

    # Define inner_echo
    def inner_echo(word1):
        """
        Concatenate n copies of word1
        :param word1:
        :return:
        """
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return inner_echo


# Call echo: twice
twice = echo(2)

# Call echo: thrice
thrice = echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))


# The keyword nonlocal and nested functions
# Define echo_shout()
def echo_shout(word):
    """
    Change the value of a nonlocal variable
    :param word:
    :return:
    """

    # Concatenate word with itself: echo_word
    echo_word = word + word

    # Print echo_word
    print(echo_word)

    # Define inner function shout()
    def shout():
        """
        Alter a variable in the enclosing scope
        :return:
        """
        # Use echo_word in nonlocal scope
        nonlocal echo_word

        # Change echo_word to echo_word concatenated with '!!!'
        echo_word = echo_word + '!!!'

    # Call function shout()
    shout()

    # Print echo_word
    print(echo_word)


# Call function echo_shout() with argument 'hello'
echo_shout('hello')


# Functions with one default argument In the previous chapter, you've learned to define functions with more than one
# parameter and then calling those functions by passing the required number of arguments. In the last video,
# Hugo built on this idea by showing you how to define functions with default arguments. You will practice that skill
# in this exercise by writing a function that uses a default argument and then calling the function a couple of times.

# Define shout_echo
def shout_echo(word1, echo=1):
    """
    Concatenate echo copies of word1 and three exclamation marks at the end of the string
    :param word1:
    :param echo:
    :return:
    """
    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word


# Call shout_echo() with "Hey": no_echo
no_echo = shout_echo("Hey")

# Call shout_echo() with "Hey" and echo=5: with_echo
with_echo = shout_echo("Hey", echo=5)

# Print no_echo and with_echo
print(no_echo)
print(with_echo)


# Functions with multiple default arguments You've now defined a function that uses a default argument - don't stop
# there just yet! You will now try your hand at defining a function with more than one default argument and then
# calling this function in various ways.
#
# After defining the function, you will call it by supplying values to all the default arguments of the function.
# Additionally, you will call the function by not passing a value to one of the default arguments - see how that
# changes the output of your function!

# Define shout_echo
def shout_echo(word1, echo=1, intense=False):
    """
    Concatenate echo copies of word1 and three exclamation marks at the end of the string
    :param word1:
    :param echo:
    :param intense:
    :return:
    """
    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Make echo_word uppercase if intense is True
    if intense:
        # Make uppercase and concatenate '!!!': echo_word_new
        echo_word_new = echo_word.upper() + '!!!'
    else:
        # Concatenate '!!!' to echo_word: echo_word_new
        echo_word_new = echo_word + '!!!'

    # Return echo_word_new
    return echo_word_new


# Call shout_echo() with "Hey", echo=5 and intense=True: with_big_echo
with_big_echo = shout_echo("Hey", echo=5, intense=True)

# Call shout_echo() with "Hey" and intense=True: big_no_echo
big_no_echo = shout_echo("Hey", intense=True)

# Print values
print(with_big_echo)
print(big_no_echo)


# Functions with variable-length arguments Flexible arguments enable you to pass a variable number of arguments to a
# function. In this exercise, you will practice defining a function that accepts a variable number of string arguments.
# The function you will define is gibberish() which can accept a variable number of string values. Its return value is
# a single string composed of all the string arguments concatenated together in the order they were passed to the
# function call. You will call the function with a single string argument and see how the output changes with another
# string argument. Recall from the previous video that, within the function definition, args is a tuple.

# Define gibberish
def gibberish(*args):
    """
    Concatenate strings in *args together
    :param args:
    :return:
    """
    # Initialize an empty string: hodgepodge
    hodgepodge = ''

    # Concatenate the strings in args
    for word in args:
        hodgepodge += word

    # Return hodgepodge
    return hodgepodge


# Call gibberish() with one string: one_word
one_word = gibberish("luke")

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)


# Functions with variable-length keyword arguments (**kwargs) Let's push further on what you've learned about
# flexible arguments - you've used *args, you're now going to use **kwargs! What makes **kwargs different is that it
# allows you to pass a variable number of keyword arguments to functions. Recall from the previous video that,
# within the function definition, kwargs is a dictionary.
#
# To understand this idea better, you're going to use **kwargs in this exercise to define a function that accepts a
# variable number of keyword arguments. The function simulates a simple status report system that prints out the
# status of a character in a movie.

# Define report_status
def report_status(**kwargs):
    """
    Print out the status of a movie character
    :param kwargs:
    :return:
    """
    print("\nBEGIN: REPORT\n")
    # Iterate over the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)
    print("\nEND REPORT")


# First call to report_status()
report_status(name="luke", affiliation="jedi", status="missing")

# Second call to report_status()
report_status(name="anakin", affiliation="sith lord", status="deceased")

# Bringing it all together (1) Recall the Bringing it all together exercise in the previous chapter where you did a
# simple Twitter analysis by developing a function that counts how many tweets are in certain languages. The output
# of your function was a dictionary that had the language as the keys and the counts of tweets in that language as
# the value.
#
# In this exercise, we will generalize the Twitter language analysis that you did in the previous chapter. You will
# do that by including a default argument that takes a column name.
#
# For your convenience, pandas has been imported as pd and the 'tweets.csv' file has been imported into the DataFrame
# tweets_df. Parts of the code from your previous work are also provided


# Import Twitter data as DataFrame: df
tweets_df = pd.read_csv('../data/tweets.csv')


# Define count_entries()
def count_entries(tweets_df, col_name='lang'):
    """
    Return a dictionary with counts of occurrences as value for each key
    :param tweets_df:
    :param col_name:
    :return:
    """
    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Extract column from DataFrame: col
    col = tweets_df[col_name]

    # Iterate over the column in DataFrame
    for entry in col:
        # If the language is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1
        # Else add the language to cols_count, set the value to 1
        else:
            cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count


# Call count_entries(): result1
result1 = count_entries(tweets_df)

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'source')

# Print result1 and result2
print(result1)
print(result2)

# Bringing it all together (2) Wow, you've just generalized your Twitter language analysis that you did in the
# previous chapter to include a default argument for the column name. You're now going to generalize this function
# one step further by allowing the user to pass it a flexible argument, that is, in this case, as many column names
# as the user would like!
#
# Once again, for your convenience, pandas has been imported as pd and the 'tweets.csv' file has been imported into
# the DataFrame tweets_df. Parts of the code from your previous work are also provided.

# Define count_entries()
def count_entries(tweets_df, *args):
    """
    Return a dictionary with counts of occurrences as value for each key
    :param tweets_df:
    :param args:
    :return:
    """
    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Iterate over column names in args
    for col_name in args:

        # Extract column from DataFrame: col
        col = tweets_df[col_name]

        # Iterate over the column in DataFrame
        for entry in col:
            # If the column is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
            # Else add the column to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang', 'source')

# Print result1 and result2
print(result1)
print(result2)
