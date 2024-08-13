import pandas as pd
from numpy.random import normal, seed
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import choice
from pandas import DataFrame

# Random walk I In the last video, you have seen how to generate a random walk of returns, and how to convert this
# random return series into a random stock price path.
#
# In this exercise, you'll build your own random walk by drawing random numbers from the normal distribution with the
# help of numpy

# Instructions 1/3
# We have already imported pandas as pd, functions normal and seed from numpy.random, and matplotlib.pyplot as plt.
#
# Set seed to 42.
# Use normal to generate 2,500 random returns with the parameters loc=.001, scale=.01 and assign this to random_walk.
# Convert random_walk to a pd.Series object and reassign it to random_walk.
# Create random_prices by adding 1 to random_walk and calculating the cumulative product.
# Multiply random_prices by 1,000 and plot the result for a price series starting at 1,000.

# Set seed (Instruction 1)
seed(42)

# Generate random walk (Instruction 2)
random_walk = normal(loc=.001, scale=.01, size=2500)

# Convert random walk to pd.Series (Instruction 3)
random_walk = pd.Series(random_walk)

# Create random prices
random_prices = random_walk.add(1).cumprod()

# Plot random prices
random_prices.mul(1000).plot()
plt.show()

# # Random walk II In the last video, you have also seen how to create a random walk of returns by sampling from actual
# # returns, and how to use this random sample to create a random stock price path.
# #
# # In this exercise, you'll build a random walk using historical returns from Facebook's stock price since IPO through
# # the end of May 31, 2017. Then you'll simulate an alternative random price path in the next exercise.
#
# # Instructions 1/3 We have already imported pandas as pd, choice and seed from numpy.random, seaborn as sns,
# # and matplotlib.pyplot as plt. We have also imported the FB stock price series since IPO in May 2012 as the variable
# # fb. Inspect this using .head().
# #
# # Set seed to 42.
# # Apply .pct_change() to generate daily Facebook returns, drop missing values, and assign to daily_returns.
# # Create a variable n_obs that contains the .count() of Facebook daily_returns.
# # Use choice() to randomly select n_obs samples from daily_returns, and assign to random_walk.
# # Convert random_walk to a pd.Series and reassign it to itself.
# # Use sns.distplot() to plot the distribution of random_walk.
#
# fb = pd.read_csv('../../data/time_series_sources/fb.csv', index_col='date', parse_dates=True)
# # set seed (Instruction 1)
# seed(42)
#
# # Calculate daily returns (Instruction 2)
# daily_returns: DataFrame = fb.pct_change().dropna()
#
# # Get number of observations (Instruction 3)
# n_obs = daily_returns.count()
#
# # Sample from daily returns (Instruction 4)
# random_walk = choice(daily_returns, size=n_obs)
#
# # Convert random walk to pd.Series (Instruction 5)
# random_walk = pd.Series(random_walk)
#
# # Plot random walk (Instruction 6)
# sns.distplot(random_walk)
# plt.show()
#
# # Random walk III In this exercise, you'll complete your random walk simulation using Facebook stock returns over the
# # last five years. You'll start off with a random sample of returns like the one you've generated during the last
# # exercise and use it to create a random stock price path.
#
# # We have already imported pandas as pd, choice and seed from numpy.random, and matplotlib.pyplot as plt. We have
# # loaded the Facebook price as a pd.DataFrame in the variable fb and a random sample of daily FB returns as pd.Series
# # in the variable random_walk.
# #
# # Select the first Facebook price by applying .first('D') to fb.price, and assign to start.
# # Add 1 to random_walk and reassign it to itself, then .append() random_walk to start and assign this to random_price.
# # Apply .cumprod() to random_price and reassign it to itself.
# # Insert random_price as new column labeled random into fb and plot the result.
#
# # Select first price (Instruction 1)
# start = fb.price.first('D')
#
# # Add 1 to random walk and append to start (Instruction 2)
# random_walk = random_walk.add(1)
# random_price = start.append(random_walk)
#
# # Calculate random prices (Instruction 3)
# random_price = random_price.cumprod()
#
# # Insert into fb and plot (Instruction 4)
# fb['random'] = random_price
# fb.plot()
# plt.show()

# Annual return correlations among several stocks
# You have seen in the video how to calculate correlations, and visualize the result.
#
# In this exercise, we have provided you with the historical stock prices for Apple (AAPL), Amazon (AMZN), IBM (IBM),
# WalMart (WMT), and ExxonMobil (XOM) for the last 4,000 trading days from July 2001 until the end of May 2017.
#
# You'll calculate the year-end returns, the pairwise correlations among all stocks, and visualize the result as an
# annotated heatmap.

# Instructions
#
# Inspect using .info(). Apply .resample() with year-end frequency (alias: 'A') to data and select the .last() price
# from each sub-period; assign this to annual_prices. Calculate annual_returns by applying .pct_change() to
# annual_prices. Calculate correlations by applying .corr() to annual_returns and print the result. Visualize
# correlations as an annotated sns.heatmap()

# Inspect data here
data = pd.read_csv('../../data/time_series_sources/five_stocks_data.csv', index_col='Date', parse_dates=True)
print(data.info())

# Calculate year-end prices here
annual_prices = data.resample('A').last()

# Calculate annual returns here
annual_returns = annual_prices.pct_change()

# Calculate and print the correlation matrix here
correlations = annual_returns.corr()
print(correlations)

# Visualize the correlations as heatmap here
sns.heatmap(correlations, annot=True)
plt.show()

# Explore and clean company listing information To get started with the construction of a market-value based index,
# you'll work with the combined listing info for the three largest US stock exchanges, the NYSE, the NASDAQ and the
# AMEX.
#
# In this and the next exercise, you will calculate market-cap weights for these stocks.
#
# We have already imported pandas as pd, and loaded the listings data set with listings information from the NYSE,
# NASDAQ, and AMEX. The column 'Market Capitalization' is already measured in USD mn.

# Instructions
# Inspect listings using .info().
# Move the column 'Stock Symbol' into the index (inplace).
# Drop all companies with missing 'Sector' information from listings.
# Select companies with IPO Year before 2019.
# Inspect the result of the changes you just made using .info().
# Show the number of companies per 'Sector' using .groupby() and .size(). Sort the output in descending order.

# Inspect listings (Instruction 1)
listings = pd.read_csv('../../data/time_series_sources/listings_1.csv')
print(listings.info())

# Move 'Stock Symbol' into the index (Instruction 2)
listings.set_index('Stock Symbol', inplace=True)

# Drop rows with missing 'Sector' data (Instruction 3)
listings.dropna(subset=['Sector'], inplace=True)

# Select companies with IPO Year before 2019 (Instruction 4)
listings = listings[listings['IPO Year'] < 2019]

# Inspect the new listings data (Instruction 5)
print(listings.info())

# Show the number of companies per Sector (Instruction 6)
print(listings.groupby('Sector').size().sort_values(ascending=False))

# Select and inspect index components Now that you have imported and cleaned the listings data, you can proceed to
# select the index components as the largest company for each sector by market capitalization.
#
# You'll also have the opportunity to take a closer look at the components, their last market value, and last price.

# We have already imported pandas as pd, and loaded the listings data with the modifications you made during the last
# exercise.
#
# Use .groupby() and .nlargest() to select the largest company by 'Market Capitalization' for each 'Sector',
# and assign the result to components. Print components, sorted in descending order by market cap. Select Stock
# Symbol from the index of components, assign it to tickers and print the result. Create a list info_cols that holds
# the column names Company Name, Market Capitalization, and Last Sale. Next, use .loc[] with tickers and info_cols to
# print() more details about the listings sorted in descending order by Market Capitalization.

listings = pd.read_csv('../../data/time_series_sources/listings_2.csv', index_col=0)

# Select largest company for each sector (Instruction 1)
components = listings.groupby('Sector')['Market Capitalization'].nlargest(1).sort_values(ascending=False)

# Print components, sorted by market cap (Instruction 2)
print(components)

# Select stock symbols and print the result (Instruction 3)
tickers = components.index.get_level_values('Stock Symbol')
# get_level_values() is used to get the values of the index
print(tickers)

# Print company name, market cap, and last price for each component (Instruction 4)
info_cols = ['Company Name', 'Market Capitalization', 'Last Sale']
print(listings.loc[tickers, info_cols].sort_values('Market Capitalization', ascending=False))

# Import index component price information Now you'll use the stock symbols for the companies you selected in the
# last exercise to calculate returns for each company.

# We have already imported pandas as pd and matplotlib.pyplot as plt for you. We have also made the variable tickers
# available to you, which contains the Stock Symbol for each index component as a list.
#
# Print tickers to verify the content matches your result from the last exercise. Use pd.read_csv() to import
# 'stock_prices.csv', parsing the 'Date' column and also setting the 'Date' column as index before assigning the
# result to stock_prices. Inspect the result using .info(). Calculate the price return for the index components by
# dividing the last row of stock_prices by the first, subtracting 1 and multiplying by 100. Assign the result to
# price_return. Plot a horizontal bar chart of the sorted returns with the title Stock Price Returns.

# Print tickers (Instruction 1)
tickers = ['RIO', 'ILMN', 'CPRT', 'EL', 'AMZN', 'PAA', 'GS', 'AMGN', 'MA', 'TEF', 'AAPL', 'UPS']
print(tickers)

# Import stock prices and inspect the result (Instruction 2)
stock_prices = pd.read_csv('../../data/time_series_sources/stock_prices.csv', parse_dates=['Date'], index_col='Date')
print(stock_prices.info())

# Calculate the returns (Instruction 3)
price_return = stock_prices.iloc[-1].div(stock_prices.iloc[0]).sub(1).mul(100)

# Plot horizontal bar chart (Instruction 4)
price_return.sort_values().plot(kind='barh', title='Stock Price Returns')
plt.show()
