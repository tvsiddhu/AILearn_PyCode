import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

pd.options.mode.chained_assignment = None

# 1. Getting to know your data

# 1.1 Import so_survey_csv into so_survey_df
so_survey_df = pd.read_csv("../../../data/preprocessing_data_sources/Combined_DS_v10.csv")

# 1.2 Print the first five rows of the DataFrame
print("**********************\nPrint first five rows\n**********************\n", so_survey_df.head(5))

# 1.3 Print the data type of each column
print("**********************\nPrint data types\n**********************\n", so_survey_df.dtypes)

# 1.4 Create subset of only the numeric columns
so_numeric_df = so_survey_df.select_dtypes(include=['int', 'float'])

# 1.5 Print the column names contained in so_survey_df_num
print("**********************\nPrint column names\n**********************\n", so_numeric_df.columns)

# 1.6 Convert the Country column to a one hot encoded Data Frame
one_hot_encoded = pd.get_dummies(so_survey_df, columns=['Country'], prefix='OH')

# 1.7 Print the columns names of the one hot encoded DataFrame
print("**********************\nPrint one-hot encoded column names\n**********************\n", one_hot_encoded.columns)

# 1.8 Create dummy variables for the Country column
dummy = pd.get_dummies(so_survey_df, columns=['Country'], drop_first=True, prefix='DM')

# 1.9 Print the columns names of the dummy DataFrame
print("**********************\nPrint column names of dummy DF\n**********************\n", dummy.columns)

# 1.10 Dealing with uncommon categories
# Create a series out of the Country column
countries = so_survey_df['Country']

# Get the counts of each category
country_counts = countries.value_counts()

# Print the count values for each category
print("**********************\nPrint count values for each category\n**********************\n", country_counts)

# 1.11 Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)

# Print the top 5 rows in the mask series
print("**********************\nPrint top 5 rows in mask series\n**********************\n", mask.head())

# 1.12 Label categories that occur less than 10 times as 'Other'
countries[mask] = 'Other'

# Print the updated category counts
print("**********************\nPrint updated category counts\n**********************\n", countries.value_counts())

# 1.13 Binarizing columns
# Create the Paid_Job column filled with zeros
so_survey_df['Paid_Job'] = 0

# Replace all the Paid_Job values where ConvertedSalary is > 0
so_survey_df.loc[so_survey_df['ConvertedSalary'] > 0, 'Paid_Job'] = 1

# Print the first five rows of the columns
print("**********************\nPrint first five rows\n**********************\n",
      so_survey_df[['Paid_Job', 'ConvertedSalary']].head())

# 1.14 Binning values
# Bin the continuous variable ConvertedSalary into 5 bins
so_survey_df['equal_binned'] = pd.cut(so_survey_df['ConvertedSalary'], 5)

# Print the first 5 rows of the equal_binned column
print("**********************\nPrint first 5 rows of equal binned columns\n**********************\n",
      so_survey_df[['equal_binned', 'ConvertedSalary']].head())

# Specify the boundaries of the bins
bins = [-np.inf, 10000, 50000, 100000, 150000, np.inf]

# Bin labels
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

# Bin the continuous variable ConvertedSalary using these boundaries
so_survey_df['boundary_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins=bins, labels=labels)

# Print the first 5 rows of the boundary_binned column
print("**********************\nPrint first 5 rows of boundary_binned columns\n**********************\n",
      so_survey_df[['boundary_binned', 'ConvertedSalary']].head())

# 1.15 How sparse is my data?
# Subset the DataFrame
sub_df = so_survey_df[['Age', 'Gender']]

# Print the number of non-missing values
print("**********************\nPrint non-missing values\n**********************\n", sub_df.notnull().sum())

# Print the top 10 entries of the DataFrame
print("**********************\nPrint top 10 entries\n**********************\n", sub_df.head(10))

# 1.16 Finding the missing values
# Print the locations of the missing values in the first 10 rows
print("**********************\nPrint locations of missing values\n**********************\n", sub_df.head(10).isnull())

# Print the locations of the non-missing values in the first 10 rows
print("**********************\nPrint locations of non-missing values\n**********************\n",
      sub_df.head(10).notnull())

# 1.17 Listwise deletion
# Print the number of rows and columns
print("**********************\nPrint number of rows and columns\n**********************\n", so_survey_df.shape)

# Create a new DataFrame dropping all incomplete rows
no_missing_values_rows = so_survey_df.dropna()

# Print the shape of the new DataFrame
print("**********************\nPrint shape of new DataFrame\n**********************\n", no_missing_values_rows.shape)

# Create a new DataFrame dropping all columns with incomplete rows
no_missing_values_cols = so_survey_df.dropna(axis=1)

# Print the shape of the new DataFrame
print("**********************\nPrint shape of new DataFrame after dropping incomplete rows\n**********************\n",
      no_missing_values_cols.shape)

# Drop all rows where Gender is missing
no_gender = so_survey_df.dropna(subset=['Gender'])

# Print the shape of the new DataFrame
print(
    "**********************\nPrint shape of new DataFrame after dropping rows where Gender is "
    "missing\n****************\n",
    no_gender.shape)

# 1.18 Replacing missing values with constants
# Print the count of occurrences
print(
    "**********************\nPrint count of occurrences after replacing values with constants\n**********************\n"
    , so_survey_df['Gender'].value_counts())

# Replace missing values in the Gender column with 'Not Given'
so_survey_df = so_survey_df.fillna({'Gender': 'Not Given'})

print(
    "**********************\nPrint count of occurrences after replacing missing values in Gender column with Not "
    "Given\n**********************\n",
    so_survey_df['Gender'].value_counts())

# 1.19 Filling continuous missing values
# Print the first five rows of StackOverflowJobsRecommend column
print("**********************\nPrint first five rows of StackOverflowJobsRecommend column\n**********************\n",
      so_survey_df['StackOverflowJobsRecommend'].head())

# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'] = so_survey_df['StackOverflowJobsRecommend'].fillna(
    so_survey_df['StackOverflowJobsRecommend'].mean())

# Print the first five rows of StackOverflowJobsRecommend column
print(
    "**********************\nPrint first five rows of StackOverflowJobsRecommend column after replacing Nan with "
    "mean\n**********************\n",
    so_survey_df['StackOverflowJobsRecommend'].head())

# Round the StackOverflowJobsRecommend values
so_survey_df['StackOverflowJobsRecommend'] = round(so_survey_df['StackOverflowJobsRecommend'])

# Print the top 5 rows
print("**********************\nPrint top 5 rows of StackOverflowJobsRecommend column after rounding the values "
      "\n**********************\n", so_survey_df['StackOverflowJobsRecommend'].head())

# 1.20 Dealing with stray characters (I)
# Remove the commas in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(',', '')

# Remove the dollar signs in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('$', '')

# 1.21 Dealing with stray characters (II)
# Attempt to convert the column to numeric values
numeric_vals = pd.to_numeric(so_survey_df['RawSalary'], errors='coerce')

# Find the indexes of missing values
idx = numeric_vals.isna()

# Print the relevant rows
print("**********************\nPrint relevant rows with missing values\n**********************\n",
      so_survey_df['RawSalary'][idx])

# Replace the offending characters
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype(str).str.replace('£', '')

# Convert the column to float
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype('float')

# Print the column
print("**********************\nPrint column after replacing the offending characters and converting the "
      "column\n**********************\n", so_survey_df['RawSalary'])

# 1.22 Method chaining
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype(str).str.replace(',', '').str.replace('$', '').str.replace(
    '£', '').astype('float')

# Print the RawSalary column
print("**********************\nPrint RawSalary column after method chaining\n**********************\n",
      so_survey_df['RawSalary'])

# 1.23 Data distributions
# Create a histogram
print("**********************\nCreate histogram of the dataframe\n**********************\n")
so_numeric_df.hist()
plt.show()

# Create a boxplot of two columns
print("**********************\nCreate boxplot of Age and Years Experience columns\n**********************\n")
so_numeric_df[['Age', 'Years Experience']].boxplot()
plt.show()

# Create a boxplot of ConvertedSalary
print("**********************\nCreate boxplot of ConvertedSalary column\n**********************\n")
so_numeric_df[['ConvertedSalary']].boxplot()
plt.show()

# Plot pairwise relationships
print("**********************\nPlot pairwise relationships\n**********************\n")
sns.pairplot(so_numeric_df)
plt.show()

# Print summary statistics
print("**********************\nPrint summary statistics\n**********************\n", so_numeric_df.describe())

# 1.24 Normalization

# Initialize MinMaxScaler
MM_scaler = MinMaxScaler()

# Fit MM_scaler to the data
MM_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_MM'] = MM_scaler.transform(so_numeric_df[['Age']])

# Compare the original and transformed column
print("**********************\nCompare original and transformed column\n**********************\n",
      so_numeric_df[['Age_MM', 'Age']].head())

# 1.25 Standardization
# Initialize StandardScaler
SS_scaler = StandardScaler()

# Fit SS_scaler to the data
SS_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_SS'] = SS_scaler.transform(so_numeric_df[['Age']])

# Compare the original and transformed column
print("**********************\nCompare original and transformed column\n**********************\n",
      so_numeric_df[['Age_SS', 'Age']].head())

# 1.26 Log transformation

# Initialize PowerTransformer
pow_trans = PowerTransformer()

# Train the transform on the data
pow_trans.fit(so_numeric_df[['ConvertedSalary']])

# Apply the power transform to the data
so_numeric_df['ConvertedSalary_LG'] = pow_trans.transform(so_numeric_df[['ConvertedSalary']])

# Plot the data before and after the transformation
so_numeric_df[['ConvertedSalary', 'ConvertedSalary_LG']].hist()
plt.show()

# 1.27 Percentage based outlier removal
# Find the 95th quantile
quantile = so_numeric_df['ConvertedSalary'].quantile(0.95)

# Trim the outliers
trimmed_df = so_numeric_df[so_numeric_df['ConvertedSalary'] < quantile]

# The original histogram
so_numeric_df[['ConvertedSalary']].hist()
plt.show()
plt.clf()

# The trimmed histogram
trimmed_df[['ConvertedSalary']].hist()
plt.show()

# 1.28 Statistical outlier removal
# Find the mean and standard dev
std = so_numeric_df['ConvertedSalary'].std()
mean = so_numeric_df['ConvertedSalary'].mean()

# Calculate the cutoff
cut_off = std * 3
lower, upper = mean - cut_off, mean + cut_off

# Trim the outliers
trimmed_df = so_numeric_df[(so_numeric_df['ConvertedSalary'] < upper) & (so_numeric_df['ConvertedSalary'] > lower)]

# The trimmed box plot
so_numeric_df[['ConvertedSalary']].boxplot()
plt.show()

# 1.29 Train and testing transformations (I)
# Apply a standard scaler to the data
SS_scaler = StandardScaler()

# Fit the standard scaler to the data
SS_scaler.fit(so_numeric_df[['Age']])

so_numeric_df = so_numeric_df.fillna({'Age': 0})
so_test_numeric = so_numeric_df.sample(20)
so_test_numeric = so_test_numeric.fillna({'Age': 0})

so_train_numeric = so_numeric_df.drop(so_test_numeric.index)

# Transform the test data using the fitted scaler
so_test_numeric['Age_ss'] = SS_scaler.transform(so_test_numeric[['Age']])
print("**********************\nPrint transformed test data\n**********************\n",
      so_test_numeric[['Age', 'Age_ss']].head())

# Apply a standard scaler to the data
SS_scaler = StandardScaler()

# Fit the standard scaler to the data
SS_scaler.fit(so_train_numeric[['Age']])

# Transform the test data using the fitted scaler
so_test_numeric['Age_ss'] = SS_scaler.transform(so_test_numeric[['Age']])
print("**********************\nPrint transformed test data\n**********************\n",
      so_test_numeric[['Age', 'Age_ss']].head())

# 1.30 Train and testing transformations (II)

train_std = so_train_numeric['ConvertedSalary'].std()
train_mean = so_train_numeric['ConvertedSalary'].mean()

cut_off = train_std * 3
train_lower, train_upper = train_mean - cut_off, train_mean + cut_off

# Trim the test DataFrame
trimmed_df = so_test_numeric[
    (so_test_numeric['ConvertedSalary'] < train_upper) & (so_test_numeric['ConvertedSalary'] > train_lower)]
print("**********************\nPrint trimmed test DataFrame\n**********************\n", trimmed_df)

# 1.31 Cleaning up your text

speech_df = pd.read_csv("../../../data/preprocessing_data_sources/inaugural_speeches.csv")

# Print the first 5 rows of the text column
print("**********************\nPrint first 5 rows of text column\n**********************\n", speech_df['text'].head())

# Replace all non letter characters with a whitespace
speech_df['text_clean'] = speech_df['text'].str.replace('[^a-zA-Z]', ' ')

# Change to lower case
speech_df['text_clean'] = speech_df['text_clean'].str.lower()

# Print the first 5 rows of the text_clean column
print("**********************\nPrint first 5 rows of text_clean column\n**********************\n",
      speech_df['text_clean'].head())

# 1.32 High level text features
# Find the length of each text
speech_df['char_cnt'] = speech_df['text_clean'].str.len()

# Count the number of words in each text
speech_df['word_cnt'] = speech_df['text_clean'].str.split().str.len()

# Find the average length of word
speech_df['avg_word_length'] = speech_df['char_cnt'] / speech_df['word_cnt']

# Print the first 5 rows of these columns
print("**********************\nPrint first 5 rows of char_cnt, word_cnt and avg_word_length "
      "columns\n**********************\n",
      speech_df[['char_cnt', 'word_cnt', 'avg_word_length']].head())

# 1.33 Counting words (I)

# Instantiate CountVectorizer
cv = CountVectorizer()

# Fit the vectorizer
cv.fit(speech_df['text_clean'])

# Print feature names
print("**********************\nPrint feature names\n**********************\n", cv.get_feature_names_out())

# 1.34 Counting words (II)
# Apply the vectorizer
cv_transformed = cv.transform(speech_df['text_clean'])

# Print the full array
cv_array = cv_transformed.toarray()
print("**********************\nPrint full array\n**********************\n", cv_array)

# Print the shape of cv_array
print("**********************\nPrint shape of cv_array\n**********************\n", cv_array.shape)

# 1.35 Limiting your features
# Specify arguements to limit the number of features generated
cv = CountVectorizer(min_df=0.2, max_df=0.8)

# Fit, transform, and convert into array
cv_transformed = cv.fit_transform(speech_df['text_clean'])
cv_array = cv_transformed.toarray()

# Print the array shape
print("**********************\nPrint array shape\n**********************\n", cv_array.shape)

# 1.36 Text to DataFrame
# Create a DataFrame with these features
cv_df = pd.DataFrame(cv_array, columns=cv.get_feature_names_out()).add_prefix('Counts_')

# Add the new columns to the original DataFrame
speech_df_new = pd.concat([speech_df, cv_df], axis=1, sort=False)

# Print the first 5 rows of the new DataFrame
print("**********************\nPrint first 5 rows of new DataFrame\n**********************\n",
      speech_df_new.head())

# 1.37 erm frequency-inverse document frequency (Tf-idf)

# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features=100, stop_words='english')

# Fit the vectorizer and transform the data
tv_transformed = tv.fit_transform(speech_df['text_clean'])

# Create a DataFrame with these features
tv_df = pd.DataFrame(tv_transformed.toarray(), columns=tv.get_feature_names_out()).add_prefix('TFIDF_')

# Print the first 5 rows of the DataFrame
print("**********************\nPrint first 5 rows of DataFrame\n**********************\n", tv_df.head())

# 1.38 Inspecting Tf-idf values
# Isolate the row to be examined
sample_row = tv_df.iloc[0]

# Print the top 5 words of the sorted output
print("**********************\nPrint top 5 words of the sorted output\n**********************\n",
      sample_row.sort_values(ascending=False).head())

# 1.39 Transforming unseen data

# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features=100, stop_words='english')

# Split the data into training and testing sets
train_speech_df = speech_df.iloc[:45]
test_speech_df = speech_df.iloc[45:]

# Fit the vectorizer and transform the training data
tv_transformed = tv.fit_transform(train_speech_df['text_clean'])

# Transform the test data
test_tv_transformed = tv.transform(test_speech_df['text_clean'])

# Create new features for the test set
test_tv_df = pd.DataFrame(test_tv_transformed.toarray(), columns=tv.get_feature_names_out()).add_prefix('TFIDF_')

# Print the first 5 rows of the DataFrame
print("**********************\nPrint first 5 rows of DataFrame\n**********************\n", test_tv_df.head())

# 1.40 N-grams
# Instantiate a trigram vectorizer
cv_trigram_vec = CountVectorizer(max_features=100, stop_words='english', ngram_range=(3, 3))

# Fit and apply trigram vectorizer
cv_trigram = cv_trigram_vec.fit_transform(speech_df['text_clean'])

# Print the trigram features
print("**********************\nPrint trigram features\n**********************\n", cv_trigram_vec.get_feature_names_out())

# 1.41 Finding the most common words
# Create a DataFrame of the features
cv_tri_df = pd.DataFrame(cv_trigram.toarray(), columns=cv_trigram_vec.get_feature_names_out()).add_prefix('Counts_')

# Print the top 5 words in the sorted output
print("**********************\nPrint top 5 words in the sorted output\n**********************\n",
      cv_tri_df.sum().sort_values(ascending=False).head())