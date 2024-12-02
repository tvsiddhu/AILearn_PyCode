import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

# Load the results dataset
toto_df = pd.read_csv('../../data/9.supervised_learning/toto_numbers_dates.csv')

# Drop 'Draw' and 'Date' columns
toto_df = toto_df.drop(columns=['Draw', 'Date'])

# Calculate the frequency of each number in each column
frequency_df = toto_df.apply(pd.Series.value_counts).fillna(0)

# Create a heatmap plot
plt.figure(figsize=(12, 8))
sns.heatmap(frequency_df, annot=True, cmap='viridis', yticklabels=range(1, 50))
plt.title('Frequency of Numbers in Each Column')
plt.xlabel('Number Columns')
plt.ylabel('Numbers')
plt.show()

# Print the most common number for each column
most_common_numbers = frequency_df.idxmax()
print("Most common numbers for each column:")
print(most_common_numbers)
