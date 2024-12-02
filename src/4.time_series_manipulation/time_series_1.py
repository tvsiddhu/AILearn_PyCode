import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
import numpy as np

# Your first time series
# You have learned in the video how to create a sequence of dates using pd.date_range(). You have also seen that each
# date in the resulting pd.DatetimeIndex is a pd.Timestamp with various attributes that you can access to obtain
# information about the date.
# Now, you'll create a week of data, iterate over the result, and obtain the dayofweek and day_name() for each date.

# Create the range of dates here
seven_days = pd.date_range(start='2017-1-1', periods=7)

# Iterate over the dates and print the number and name of the weekday
for day in seven_days:
    print(day.dayofweek, day.day_name())

# Create a time series of air quality data You have seen in the video how to deal with dates that are not in the
# correct format, but instead are provided as string types, represented as dtype object in pandas.
#
# We have prepared a data set with air quality data (ozone, pm25, and carbon monoxide for NYC, 2000-2017) for you to
# practice the use of pd.to_datetime()

print("------Create a time series of air quality data -----------")

data = pd.read_csv('../../data/4.time_series_sources/air_quality.csv')

# Inspect data
print(data.info())

# Convert the date column to datetime64
data['date'] = pd.to_datetime(data['date'])

# Set date column as index
data.set_index('date', inplace=True)

# Inspect data
print(data.info())

# Plot data
data.plot(subplots=True)
plt.show()

# Compare annual stock price trends
print("----------Compare annual stock price trends------------")
# create dataframe prices here
prices: DataFrame = pd.DataFrame()
yahoo: DataFrame = pd.read_csv('../../data/4.time_series_sources/yahoo_prices.csv',
                               index_col='date', parse_dates=True)

# Select data for each year and concatenate with prices here
for year in ['2013', '2014', '2015']:
    price_per_year = yahoo.loc[yahoo.index.year == int(year), ['price']].reset_index(drop=True)
    assert isinstance(price_per_year, object)
    price_per_year.rename(columns={'price': year}, inplace=True)
    prices = pd.concat([prices, price_per_year], axis=1)

# Plot prices
prices.plot()
plt.show()

# Set and change time series frequency
# Now, you'll use data on the daily carbon monoxide concentration in NYC, LA and Chicago from 2005-17.
#
# You'll set the frequency to calendar daily and then resample to monthly frequency, and visualize both series to see
# how the different frequencies affect the data.
print("--------Set and change time series frequency-----------")

co = pd.read_csv('../../data/4.time_series_sources/co_cities.csv', parse_dates=['date'], index_col='date')

# Inspect data
print(co.info())

# Set the frequency to calendar daily
co = co.asfreq('D')

# Plot the data
co.plot(subplots=True)
plt.show()

# Set the frequency to monthly
co = co.asfreq('ME')

# Plot the data
co.plot(subplots=True)
plt.show()

# Shifting stock prices across time The first method to manipulate time series that you saw in the video was .shift(
# ), which allows you shift all values in a Series or DataFrame by a number of periods to a different time along the
# DateTimeIndex.
#
# Let's use this to visually compare a stock price series for Google shifted 90 business days into both past and future.

print("---------Shifting stock prices across time----------")

# Import data here
google = pd.read_csv('../../data/4.time_series_sources/google.csv', parse_dates=['Date'], index_col='Date')

# Set data frequency to business daily
google = google.asfreq('B')

# Create 'lagged' and 'shifted' series
google['lagged'] = google['Close'].shift(periods=-90)
google['shifted'] = google['Close'].shift(periods=90)

# Plot the Google price series
google.plot(subplots=True)
plt.show()

# Calculating stock price changes You have learned in the video how to calculate returns using current and shifted
# prices as input. Now you'll practice a similar calculation to calculate absolute changes from current and shifted
# prices, and compare the result to the function .diff().

print("---------Calculating stock price changes-----------")

# Created shifted_30 here
yahoo['shifted_30'] = yahoo['price'].shift(periods=30)

# Subtract shifted_30 from price
yahoo['change_30'] = yahoo['price'] - yahoo['shifted_30']

# Get the 30-day price difference
yahoo['diff_30'] = yahoo['price'].diff(periods=30)

# Inspect the last five rows of price
print(yahoo.tail())

# Show the value_counts of the difference between change_30 and diff_30
print(yahoo['change_30'].sub(yahoo['diff_30']).value_counts())

# Plot the price series
yahoo.plot()
plt.show()

# Plotting multi-period returns The last time series method you have learned about in the video was .pct_change().
# Let's use this function to calculate returns for various calendar day periods, and plot the result to compare the
# different patterns.
#
# We'll be using Google stock prices from 2014-2016.

print("---------Plotting multi-period returns -----------")

# Create daily_return
google['daily_return'] = google.Close.pct_change(fill_method=None).mul(100)

# Create monthly_return
google['daily_return'] = google.Close.pct_change(30, fill_method=None).mul(100)

# Create annual_return
google['daily_return'] = google.Close.pct_change(360, fill_method=None).mul(100)

# Plot the result
google.plot(subplots=True)
plt.show()

# Compare the performance of several asset classes You have seen in the video how you can easily compare several time
# series by normalizing their starting points to 100, and plot the result.
#
# To broaden your perspective on financial markets, let's compare four key assets: stocks, bonds, gold, and oil.

print("----------Compare the performance of several asset classes----------")

# Import data here
prices = pd.read_csv('../../data/4.time_series_sources/asset_classes.csv',
                     parse_dates=['DATE'], index_col='DATE')

# Inspect prices here
print(prices.info())

# Select first prices
first_prices = prices.iloc[0]

# Create normalized
normalized = prices.div(first_prices).mul(100)

# Plot normalized
normalized.plot()
plt.show()

# Comparing stock prices with a benchmark You also learned in the video how to compare the performance of various
# stocks against a benchmark. Now you'll learn more about the stock market by comparing the three largest stocks on
# the NYSE to the Dow Jones Industrial Average, which contains the 30 largest US companies.

# Import stock prices and index here

print("--------Comparing stock prices with a benchmark----------")

stocks = pd.read_csv('../../data/4.time_series_sources/nyse.csv',
                     parse_dates=['date'], index_col='date')
dow_jones = pd.read_csv('../../data/4.time_series_sources/dow_jones.csv',
                        parse_dates=['date'], index_col='date')

# Concatenate stocks and index here
data = pd.concat([stocks, dow_jones], axis=1)
print(data.info())

# Normalize and plot your data here
data.div(data.iloc[0]).mul(100).plot()
plt.show()

# Plot performance difference vs benchmark index In the video, you learned how to calculate and plot the performance
# difference of a stock in percentage points relative to a benchmark index.
#
# Let's compare the performance of Microsoft (MSFT) and Apple (AAPL) to the S&P 500 over the last 10 years.

print("--------Plot performance difference vs benchmark index-----------")

# Create tickers
tickers = ['MSFT', 'AAPL']

# Import stock data here
stocks = pd.read_csv('../../data/4.time_series_sources/msft_aapl.csv',
                     parse_dates=['date'], index_col='date')

# Import index here
sp500 = pd.read_csv('../../data/4.time_series_sources/sp500.csv',
                    parse_dates=['date'], index_col='date')

# Concatenate stocks and index here
data = pd.concat([stocks, sp500], axis=1).dropna()

# Normalize data
normalized = data.div(data.iloc[0]).mul(100)

# Subtract the normalized index from the normalized stock prices, and plot the result
normalized[tickers].sub(normalized['SP500'], axis=0).plot()
plt.show()

# Convert monthly to weekly data You have learned in the video how to use .reindex() to conform an existing time
# series to a DateTimeIndex at a different frequency.
#
# Let's practice this method by creating monthly data and then converting this data to weekly frequency while
# applying various fill logic options.

# Set start and end dates

print("-----------Convert monthly to weekly data----------")
start = '2016-1-1'
end = '2016-2-29'

# Create monthly dates here
monthly_dates = pd.date_range(start=start, end=end, freq='ME')

# Create and print monthly here
monthly: Series = pd.Series(data=[1, 2], index=monthly_dates)
print(monthly)

# Create weekly dates here
weekly_dates = pd.date_range(start=start, end=end, freq='W')

# Print monthly, re-indexed using weekly_dates
print(monthly.reindex(weekly_dates))
print(monthly.reindex(weekly_dates, method='bfill'))
print(monthly.reindex(weekly_dates, method='ffill'))

print("-----------Print monthly, re-indexed using weekly_dates-----------")

# Import data here
data = pd.read_csv('../../data/4.time_series_sources/unemployment.csv',
                   parse_dates=['date'], index_col='date')

# Show first five rows of weekly series
print(data.asfreq('W').head())

# Show first five rows of weekly series with bfill option
print(data.asfreq('W', method='bfill').head())

# Create weekly series with ffill option and show first five rows
weekly_ffill = data.asfreq('W', method='ffill')
print(weekly_ffill.head())

# Plot weekly_fill starting 2015 here
weekly_ffill['2015':].plot()
plt.show()

# Use interpolation to create weekly employment data You have recently used the civilian US unemployment rate,
# and converted it from monthly to weekly frequency using simple forward or backfill methods.
#
# Compare your previous approach to the new .interpolate() method that you learned about in this video.

# print("---------------------------------") # Import data here monthly = pd.read_csv(
# '../../data/4.time_series_sources/monthly_unemployment.csv', parse_dates=['date'], index_col='date')
#
# # Inspect data here
# print(monthly.info())
#
# # Create weekly dates
# weekly_dates = pd.date_range(monthly.index.min(), monthly.index.max(), freq='W')
#
# # Reindex monthly to weekly data
# weekly = monthly.reindex(weekly_dates)
#
# # Create ffill and interpolated columns
# weekly['ffill'] = weekly.UNRATE.ffill()
# weekly['interpolated'] = weekly.UNRATE.interpolate()
#
# # Plot weekly
# weekly.plot()
# plt.show()

# Interpolate debt/GDP and compare to unemployment Since you have learned how to interpolate time series, you can now
# apply this new skill to the quarterly debt/GDP series, and compare the result to the monthly unemployment rate.

print("-------Interpolate debt/GDP and compare to unemployment----------")

# Import and inspect data here
data = pd.read_csv('../../data/4.time_series_sources/debt_unemployment.csv', parse_dates=['date'], index_col='date')
print(data.info())

# Interpolate and inspect here
interpolated = data.interpolate()
print(interpolated.info())

# Plot interpolated data here
interpolated.plot(secondary_y='Unemployment')
plt.show()

# Compare weekly, monthly and annual ozone trends for NYC & LA
# You have seen in the video how to downsample and aggregate time series on air quality.
#
# First, you'll apply this new skill to ozone data for both NYC and LA since 2000 to compare the air quality trend at
# weekly, monthly and annual frequencies and explore how different resampling periods impact the visualization.

print("-------Compare weekly, monthly and annual ozone trends for NYC & LA-------------")

# Import and inspect data here
ozone = pd.read_csv('../../data/4.time_series_sources/ozone.csv', parse_dates=['date'], index_col='date')
print(ozone.info())

# Calculate and plot the weekly average ozone trend
ozone.resample('W').mean().plot(subplots=True)
plt.show()

# Calculate and plot the monthly average ozone trend
ozone.resample('ME').mean().plot(subplots=True)
plt.show()

# Calculate and plot the annual average ozone trend
ozone.resample('YE').mean().plot(subplots=True)
plt.show()

# Compare monthly average stock prices for Facebook and Google Now, you'll apply your new resampling skills to daily
# stock price series for Facebook and Google for the 2015-2016 period to compare the trend of the monthly averages.

print("---------Compare monthly average stock prices for Facebook and Google-----------")

# Import and inspect data here
stocks = pd.read_csv('../../data/4.time_series_sources/stocks.csv', parse_dates=['date'], index_col='date')
print(stocks.info())

# Calculate and plot the monthly averages
monthly_average = stocks.resample('ME').mean()
monthly_average.plot(subplots=True)
plt.show()

# Compare quarterly GDP growth rate and stock returns With your new skill to downsample and aggregate time series,
# you can compare higher-frequency stock price series to lower-frequency economic time series.
#
# As a first example, let's compare the quarterly GDP growth rate to the quarterly rate of return on the (resampled)
# Dow Jones Industrial index of 30 large US stocks.
#
# GDP growth is reported at the beginning of each quarter for the previous quarter. To calculate matching stock
# returns, you'll resample the stock index to quarter start frequency using the alias 'QS', and aggregating using the
# .first() observations.

print("---------Compare quarterly GDP growth rate and stock returns---------")

# Import and inspect gdp_growth here
gdp_growth = pd.read_csv('../../data/4.time_series_sources/gdp_growth.csv', parse_dates=['date'], index_col='date')
print(gdp_growth.info())

# Import and inspect djia here
djia = pd.read_csv('../../data/4.time_series_sources/djia.csv', parse_dates=['date'], index_col='date')
print(djia.info())

# Calculate djia quarterly returns here
djia_quarterly = djia.resample('QS').first()
djia_quarterly_return = djia_quarterly.pct_change().mul(100)

# Concatenate, rename and plot djia_quarterly_return and gdp_growth here
data = pd.concat([gdp_growth, djia_quarterly_return], axis=1)
data.columns = ['gdp', 'djia']
data.plot()
plt.show()

# Visualize monthly mean, median and standard deviation of S&P500 returns
# You have also learned how to calculate several aggregate statistics from upsampled data.
#
# Let's use this to explore how the monthly mean, median and standard deviation of daily S&P500 returns have trended
# over the last 10 years.

print("----------Visualize monthly mean, median and standard deviation of S&P500 returns-----------")

# Import data here
sp500 = pd.read_csv('../../data/4.time_series_sources/sp500.csv', parse_dates=['date'], index_col='date')

# Calculate daily returns here
daily_returns = sp500.squeeze().pct_change()

# Resample and calculate statistics
stats = daily_returns.resample('ME').agg(['mean', 'median', 'std'])

# Plot stats here
stats.plot()
plt.show()

# Rolling average air quality since 2010 for new york city The last video was about rolling window functions. To
# practice this new tool, you'll start with air quality trends for New York City since 2010. In particular,
# you'll be using the daily Ozone concentration levels provided by the Environmental Protection Agency to calculate &
# plot the 90 and 360 day rolling average.

print("-----------Rolling average air quality since 2010 for new york city-------------")

# Import and inspect ozone data here
data = pd.read_csv('../../data/4.time_series_sources/ozone_2.csv', parse_dates=['date'], index_col='date')
print(data.info())

# Calculate 90d and 360d rolling mean for the last price
data['90D'] = data['Ozone'].rolling('90D').mean()
data['360D'] = data['Ozone'].rolling('360D').mean()

# Plot data
data['2010':].plot(title='New York City')
plt.show()

# Rolling 360-day median & std. deviation for nyc ozone data since 2000 The last video also showed you how to
# calculate several rolling statistics using the .agg() method, similar to .groupby().
#
# Let's take a closer look at the air quality history of NYC using the Ozone data you have seen before. The daily
# data are very volatile, so using a longer term rolling average can help reveal a longer term trend.
#
# You'll be using a 360-day rolling window, and .agg() to calculate the rolling mean and standard deviation for the
# daily average ozone values since 2000.

print("----------- Rolling 360-day median & std. deviation for nyc ozone data since 2000-----------")

# Import and inspect ozone data here
data = pd.read_csv('../../data/4.time_series_sources/ozone_2.csv', parse_dates=['date'], index_col='date')
data.dropna(inplace=True)

print(data.info())

# Calculate the rolling mean and std here
rolling_stats = data['Ozone'].rolling(360).agg(['mean', 'std'])

# Join rolling_stats with ozone data
stats = data.join(rolling_stats)

# Plot stats
stats.plot(subplots=True)
plt.show()

# Rolling quantiles for daily air quality in nyc You learned in the last video how to calculate rolling quantiles to
# describe changes in the dispersion of a time series over time in a way that is less sensitive to outliers than
# using the mean and standard deviation.
#
# Let's calculate rolling quantiles - at 10%, 50% (median) and 90% - of the distribution of daily average ozone
# concentration in NYC using a 360-day rolling window.

print("----------Rolling quantiles for daily air quality in nyc------------")

# Resample, interpolate and inspect ozone data here
data = pd.read_csv('../../data/4.time_series_sources/ozone_2.csv', parse_dates=['date'], index_col='date')
data = data.resample('D').interpolate()
print(data.info())

# Create the rolling window
rolling = data['Ozone'].rolling(360)

# Insert the rolling quantiles to the monthly returns
data['q10'] = rolling.quantile(0.1)
data['q50'] = rolling.quantile(0.5)
data['q90'] = rolling.quantile(0.9)

# Plot the data
data.plot()
plt.show()

# Cumulative sum vs .diff()
# In the video, you have learned about expanding windows that allow you to run cumulative calculations.
#
# The cumulative sum method has in fact the opposite effect of the .diff() method that you came across in chapter 1.
#
# To illustrate this, let's use the Google stock price time series, create the differences between prices,
# and reconstruct the series using the cumulative sum.

# print("-----------Cumulative sum vs .diff()-----------")
#
# # Import and inspect data here
# data = pd.read_csv('../../data/4.time_series_sources/google.csv', parse_dates=['Date'], index_col='Date')
# print(google.info())
#
# # Calculate differences
# differences = data.diff().dropna()
#
# # Select start price
# start_price = data.first('D')
#
# # Calculate cumulative sum
# cumulative_sum = start_price.append(differences).cumsum()
#
# # Validate cumulative sum equals google prices
# data.equals(cumulative_sum)

# Cumulative return on $1,000 invested in google vs apple I To put your new ability to do cumulative return
# calculations to practical use, let's compare how much $1,000 would be worth if invested in Google ('GOOG') or Apple
# ('AAPL') in 2010.

print("-----------Cumulative return on $1,000 invested in google vs apple I-----------")

# Import and inspect data here
data = pd.read_csv('../../data/4.time_series_sources/stocks.csv', parse_dates=['date'], index_col='date')
print(data.info())

# Define your investment
investment = 1000

# Calculate the daily returns here
returns = data.pct_change()

# Calculate the cumulative returns here
returns_plus_one = returns + 1
cumulative_return = returns_plus_one.cumprod()

# Calculate and plot the investment return here
cumulative_return.mul(investment).plot()
plt.show()

# Cumulative return on $1,000 invested in google vs apple II Apple outperformed Google over the entire period,
# but this may have been different over various 1-year sub periods, so that switching between the two stocks might
# have yielded an even better result.
#
# To analyze this, calculate that cumulative return for rolling 1-year periods, and then plot the returns to see when
# each stock was superior.

print("-----------Cumulative return on $1,000 invested in google vs apple II-----------")


# Define a multi_period_return function
def multi_period_return(period_returns):
    """
    Calculate the overall multi period return.
    :param period_returns:
    :return:
    """
    period_returns = period_returns + 1
    return np.prod(period_returns) - 1


# Calculate daily returns
daily_returns = data.pct_change()

# Calculate rolling_annual_returns
rolling_annual_returns = daily_returns.rolling('360D').apply(multi_period_return)

# Plot rolling_annual_returns
rolling_annual_returns.mul(daily_returns).plot()
plt.show()
