# Uniform currencies
# In this exercise and throughout this chapter, you will be working with a retail banking dataset stored in
# the banking DataFrame. The dataset contains data on the amount of money stored in accounts (acct_amount), their
# currency (acct_cur), amount invested (inv_amount), account opening date (account_opened), and last transaction date
# (last_transaction) that were consolidated from American and European branches.You are tasked with understanding the
# average account size and how investments vary by the size of account, however in order to produce this analysis
# accurately, you first need to unify the currency amount into dollars.

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt


# Load the dataset
banking = pd.read_csv('../data/cleaning_data_sources/banking_funds.csv')

# Find values of acct_cur that are equal to 'euro'
acct_eu = banking['acct_cur'] == 'euro'

# Convert acct_amount where it is in euro to dollars
banking.loc[acct_eu, 'acct_amount'] = banking.loc[acct_eu, 'acct_amount'] * 1.1

# Unify acct_cur column by changing 'euro' values to 'dollar'
banking.loc[acct_eu, 'acct_cur'] = 'dollar'

# Assert that only dollar currency remains
assert banking['acct_cur'].unique() == 'dollar'
print(banking.head())

# Print the header of account_opened
print(banking['account_opened'].head())

# Uniform dates
# After having unified the currencies of your different account amounts, you want to add a temporal dimension to your
# analysis and see how customers have been investing on the platform over time. The account_opened column represents
# when customers opened their accounts and is a good proxy for the real date of account creation.
# However, since this data was consolidated from multiple sources, you need to make sure that all dates are of the same
# format. You will do so by converting this column into a datetime object, while making sure that the format is inferred
# and potentially incorrect formats are set to missing.

# Print the header of account_opened
print(banking['account_opened'].head())

# Convert account_opened to datetime
banking['account_opened'] = pd.to_datetime(banking['account_opened'],
                                             # Infer datetime format
                                             infer_datetime_format=True,
                                             # Return missing value for error
                                             errors='coerce')

# Get year of account opened
banking['acct_year'] = banking['account_opened'].dt.strftime('%Y')

# Print acct_year
print(banking['acct_year'])

# How's our data integrity?
# New data has been merged into the banking DataFrame that contains details on how investments in the inv_amount
# column are allocated across four different funds A, B, C, and D. Furthermore, the age and occupation of customers
# are now stored in the age and occupation columns respectively.
# You want to understand how customers of different age groups invest. However, you want to make sure the data you're
# analyzing is correct. You will do so by cross-referencing values of inv_amount and age against the amount invested
# in different funds, and the reported occupation.

# Store fund columns to sum against
fund_columns = ['fund_A', 'fund_B', 'fund_C', 'fund_D']

# Find rows where fund_columns row sum == inv_amount
inv_equ = banking[fund_columns].sum(axis=1) == banking['inv_amount']

# Store consistent and inconsistent data
consistent_inv = banking[inv_equ]
inconsistent_inv = banking[~inv_equ]

# Store consistent and inconsistent data
print("Number of inconsistent investments: ", inconsistent_inv.shape[0])

# In this particular case, the file changes to banking_birthdates.csv. The data has been modified so that all birth_date
# values are missing. You want to get a better understanding of how much of this data is missing. This will help you
# decide what steps should be taken to further balance the dataset.



# Store today's date and find ages
today = pd.to_datetime('now')
ages_manual = today.year - banking['birth_date'].dt.year

# Find rows where age column == ages_manual
age_equ = ages_manual == banking['age']

# Store consistent and inconsistent data
consistent_ages = banking[age_equ]
inconsistent_ages = banking[~age_equ]

# Store consistent and inconsistent data
print("Number of inconsistent ages: ", inconsistent_ages.shape[0])

# Is this missing at random?
# Missing Completely at Random: No systematic relationship between a column's missing values and other or own values.
# Missing at Random: There is a systematic relationship between a column's missing values and other observed values.
# Missing not at Random: There is a systematic relationship between a column's missing values and unobserved values.

# Missing Investors Dealing with missing data is one of the most common tasks in data science. There are a variety of
# types of missingness, as well as a variety of types of solutions to missing data.
# # You just received a new version of the banking DataFrame containing data on the amount held and invested for new
# and existing customers. However, there are rows with missing inv_amount values.
# # You know for a fact that most customers below 25 do not have investment accounts yet, and suspect it could be
# driving the missingness.

# Print number of missing values in banking
print(banking.isna().sum())

# Visualize missingness matrix
msno.matrix(banking)
plt.show()

# Isolate missing and non-missing values of inv_amount
missing_investors = banking[banking['inv_amount'].isna()]
investors = banking[~banking['inv_amount'].isna()]

# Sort banking by age and visualize
banking_sorted = banking.sort_values('age')
msno.matrix(banking_sorted)
plt.show()

# Follow the money
# In this exercise, you're working with another version of the banking DataFrame that contains missing values for both
# the cust_id column and the acct_amount column.
# You want to produce analysis on how many unique customers the bank has, the average amount held by customers and more.
# You know that rows with missing cust_id don't really help you, and that on average acct_amount is usually 5 times the
# amount of inv_amount.

# Drop missing values of cust_id
banking_fullid = banking.dropna(subset=['cust_id'])

# Compute estimated acct_amount
acct_imp = banking_fullid['inv_amount'] * 5

# Impute missing acct_amount with corresponding acct_imp
banking_imputed = banking_fullid.fillna({'acct_amount': acct_imp})

# Print number of missing values
print(banking_imputed.isna().sum())




