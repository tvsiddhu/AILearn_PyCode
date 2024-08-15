import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import normal, seed
import openpyxl as px


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
annual_prices = data.resample('YE').last()

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

# Calculate number of shares outstanding
# The next step towards building a value-weighted index is to calculate the number of shares for each index component.
#
# The number of shares will allow you to calculate the total market capitalization for each component given the
# historical price series in the next exercise.

# Inspect listings and print tickers. Use .loc[] with the list of tickers to select the index components and the
# columns 'Market Capitalization' and 'Last Sale'; assign this to components. Print the first five rows of
# components. Create no_shares by dividing Market Capitalization by 'Last Sale'. Print no_shares in descending order.

# Inspect listings and print tickers (Instruction 1)
print(listings.info())
print(tickers)

# Select components and relevant columns from listings (Instruction 2)
components = listings.loc[tickers, ['Market Capitalization', 'Last Sale']]

# Print the first rows of components (Instruction 3)
print(components.head())

# Calculate the number of shares here (Instruction 4)
no_shares = components['Market Capitalization'].div(components['Last Sale'])
print(no_shares.sort_values(ascending=False))

# Print the sorted no_shares
print(no_shares.sort_values(ascending=False))

# Create time series of market value You can now use the number of shares to calculate the total market
# capitalization for each component and trading date from the historical price series.
#
# The result will be the key input to construct the value-weighted stock index, which you will complete in the next
# exercise.

# We have already imported pandas as pd and matplotlib.pyplot as plt for you. We have also created the variables
# components and stock_prices that you worked with in the last exercises.
#
# Select the 'Number of Shares' from components, assign to no_shares, and print the result, sorted in the default (
# ascending) order. Multiply stock_prices by no_shares to create a time series of market cap per ticker, and assign
# it to market_cap. Select the first and the last row of market_cap and assign these to first_value and last_value.
# Use pd.concat() to concatenate first_value and last_value along axis=1 and plot the result as horizontal bar chart.

# Select the number of shares (Instruction 1)
no_shares = components['Market Capitalization'].div(components['Last Sale'])
print(no_shares.sort_values())

# Create the series of market cap per ticker (Instruction 2)
market_cap = stock_prices.mul(no_shares)

# Select first and last market cap here
first_value = market_cap.iloc[0]
last_value = market_cap.iloc[-1]

# Concatenate and plot first and last market cap here
pd.concat([first_value, last_value], axis=1).plot(kind='barh')

# Calculate & plot the composite index By now you have all ingredients that you need to calculate the aggregate stock
# performance for your group of companies.Use the time series of market capitalization that you created in the last
# exercise to aggregate the market value for each period, and then normalize this series to convert it to an index.

# Aggregate the market cap per trading day by applying .sum() to market_cap_series with axis=1, assign to raw_index
# and print the result. Normalize the aggregate market cap by dividing by the first value of raw_index and
# multiplying by 100. Assign this to index and print the result. Plot the index with the title 'Market-Cap Weighted
# Index'.

# Aggregate and print the market cap per trading day (Instruction 1)
raw_index = market_cap.sum(axis=1)
print(raw_index)

# Normalize the aggregate market cap here (Instruction 2)
index: object = raw_index.div(raw_index.iloc[0]).mul(100)
print(index)

# Plot the index here (Instruction 3)
index.plot(title='Market-Cap Weighted Index')
plt.show()

# Calculate the contribution of each stock to the index
# You have successfully built the value-weighted index. Let's now explore how it performed over the 2010-2016 period.
# Let's also determine how much each stock has contributed to the index return.
# Divide the last index value by the first, subtract 1 and multiply by 100. Assign the result to index_return and
# print it. Select the 'Market Capitalization' column from components. Calculate the total market cap for all
# components and assign this to total_market_cap. Divide the components' market cap by total_market_cap to calculate
# the component weights, assign it to weights, and print weights with the values sorted in default (ascending) order.
# Multiply weights by the index_return to calculate the contribution by component, sort the values in ascending
# order, and plot the result as a horizontal bar chart.

# Calculate and print the index return here
index_return = (index.iloc[-1] / index.iloc[0] - 1) * 100
print(index_return)

# Select the market capitalization (Instruction 2)
market_cap = components['Market Capitalization']

# Calculate the total market cap (Instruction 3)
total_market_cap = market_cap.sum()

# Calculate the component weights, and print the result (Instruction 4)
weights = market_cap.div(total_market_cap)
print(weights.sort_values())

# Calculate and plot the contribution by component (Instruction 5)
weights.mul(index_return).sort_values().plot(kind='barh')
plt.show()

# Compare index performance against benchmark I
# You used the S&P 500 as benchmark. You can also use the Dow Jones
# Industrial Average, which contains the 30 largest stocks, and would also be a reasonable benchmark for the largest
# stocks from all sectors across the three exchanges.

# Convert index to a pd.DataFrame with the column name 'Index' and assign the result to data. Normalize djia to start
# at 100 and add it as new column to data. Show the total return for both index and djia by dividing the last row of
# data by the first, subtracting 1 and multiplying by 100. Show a plot of both of the series in data.

# Convert index series to dataframe here
data = pd.DataFrame(index=index.index, data={'Index': index})

# Normalize djia series and add as new column to data
djia = pd.read_csv('../../data/time_series_sources/djia.csv', parse_dates=['date'], index_col='date')
data['DJIA'] = djia.div(djia.iloc[0]).mul(100)

# Show total return for both index and djia
total_return = data.iloc[-1].div(data.iloc[0]).sub(1).mul(100)
print(total_return)

# Plot both index and djia
data.plot(title='Market-Cap Weighted Index vs DJIA')
plt.show()

# Compare index performance against benchmark II
# The next step in analyzing the performance of your index is to compare it against a benchmark.
#
# In the video, we have use the S&P 500 as benchmark. You can also use the Dow Jones Industrial Average,
# which contains the 30 largest stocks, and would also be a reasonable benchmark for the largest stocks from all
# sectors across the three exchanges.

# We have already imported numpy as np, pandas as pd, matplotlib.pyplot as plt for you. We have also loaded your
# Index and the Dow Jones Industrial Average (normalized) in a variable called data.
#
# Inspect data and print the first five rows. Define a function multi_period_return that takes a numpy array of
# period returns as input, and returns the total return for the period. Use the formula from the video - add 1 to the
# input, pass the result to np.prod(), subtract 1 and multiply by 100. Create a .rolling() window of length '360D'
# from data, and apply multi_period_return. Assign to rolling_return_360. Plot rolling_return_360 using the title
# 'Rolling 360D Return'.

# Inspect data
print(data.info())
print(data.head())


# Create a multi_period_return function here
def multi_period_return(period_returns):
    """

    :param period_returns:
    :return:
    """
    return (np.prod(period_returns + 1) - 1) * 100


# Calculate rolling_return_360
rolling_return_360 = data.pct_change().rolling('360D').apply(multi_period_return)

# Plot rolling_return_360
rolling_return_360.plot(title='Rolling 360D Return')
plt.show()

# Visualize your index constituent correlations
# To better understand the characteristics of your index constituents, you can calculate the return correlations.
#
# Use the daily stock prices or your index companies, and show a heatmap of the daily return correlations!

# Inspect stock_prices using .info().
# Calculate the daily returns for stock_prices and assign the result to returns.
# Calculate the pairwise correlations for returns, assign them to correlations and print the result.
# Plot a seaborn annotated heatmap of the daily return correlations with the title 'Daily Return Correlations'.

# Inspect stock_prices
print(stock_prices.info())

# Calculate the daily returns
returns = stock_prices.pct_change()

# Calculate and print the pairwise correlations
correlations = returns.corr()
print(correlations)

# Plot a heatmap of daily return correlations
sns.heatmap(correlations, annot=True)
plt.title('Daily Return Correlations')
plt.show()

# Save your analysis to multiple excel worksheets
# Now that you have completed your analysis, you may want to save all results into a single Excel workbook.
#
# Let's practice exporting various DataFrame to multiple Excel worksheets.

# Inspect both index and stock_prices using .info().
# Use .join() to combine index with stock_prices, and assign to data.
# Apply .pct_change() to data and assign to returns.
# Create pd.ExcelWriter and use with to export data and returns to excel with sheet_names of the same name.

# Inspect index and stock_prices
index.name = 'IndexName'

print(index.info())
print(stock_prices.info())

# Join index to stock_prices, and inspect the result
data = stock_prices.join(index)
print(data.info())

# Create index & stock price returns
returns = data.pct_change()

# Export data and returns to excel
with pd.ExcelWriter('../../data/time_series_sources/index_data.xlsx') as writer:
    data.to_excel(writer, sheet_name='data')
    returns.to_excel(writer, sheet_name='returns')
    