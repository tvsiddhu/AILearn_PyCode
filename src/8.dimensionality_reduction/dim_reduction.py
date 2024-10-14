import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# 1. Removing features without variance

# Load pokemon dataset
pokemon_df = pd.read_csv('../../data/dimensionality_reduction_sources/pokemon_gen1.csv')
print(pokemon_df.head())
print(pokemon_df.describe())

# Remove the feature without variance from this list
number_cols = ['HP', 'Attack', 'Defense']

# Leave this list as is for now
non_number_cols = ['Name', 'Type']

# Sub-select by combining the lists with chosen features
df_selected = pokemon_df[number_cols + non_number_cols]

# Print the first 5 rows of the new dataframe
print(df_selected.head())

# 2. Visually detecting redundant features

# Load the dataset
ansur_df_1 = pd.read_csv('../../data/dimensionality_reduction_sources/ansur_df_1.csv')
ansur_df_2 = pd.read_csv('../../data/dimensionality_reduction_sources/ansur_df_2.csv')

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_1, hue='Gender', diag_kind='hist')

# Show the plot
plt.show()

# Remove one of the redundant features
reduced_df = ansur_df_1.drop('stature_m', axis=1)

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(reduced_df, hue='Gender', diag_kind='hist')

# Show the plot
plt.show()

# Create a pairplot for ansur_df_2
sns.pairplot(ansur_df_2, hue='Gender', diag_kind='hist')

# Show the plot
plt.show()

# Remove the redundant feature
reduced_df = ansur_df_2.drop('n_legs', axis=1)

# Create a pairplot and color the points
sns.pairplot(reduced_df, hue='Gender', diag_kind='hist')

# Show the plot
plt.show()

# 3. Fitting t-SNE to the ANSUR data


# Load the dataset
ansur_df = pd.read_csv('../../data/dimensionality_reduction_sources/ansur_t-SNE.csv')

# Non-numerical columns in the dataset
non_numeric = ['Branch', 'Gender', 'Component']

# Drop the non-numerical columns from df
df_numeric = ansur_df.drop(non_numeric, axis=1)

# Create a t-SNE model with learning rate 50
m = TSNE(learning_rate=50)

# Fit and transform the t-SNE model on the numeric dataset
tsne_features = m.fit_transform(df_numeric)
print("t-SNE Features Shape:", tsne_features.shape)

# 4. t-SNE visualisation of dimensionality

# Color the points according to Army Component
sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=ansur_df['Component'])

# Show the plot
plt.show()

# Color the points by Army Branch
sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=ansur_df['Branch'])

# Show the plot
plt.show()

# Color the points by Gender
sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=ansur_df['Gender'])

# Show the plot
plt.show()

# The curse of dimensionality

# 5. Train-test split

# Load the dataset
ansur_df = pd.read_csv('../../data/dimensionality_reduction_sources/ansur_test_train.csv')

# Select the Gender column as the feature to be predicted (y)
y = ansur_df['Gender']

# Remove the Gender column to create the training data
X = ansur_df.drop('Gender', axis=1)

# Perform a 70% train and 30% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(f"{X_test.shape[0]} rows in test set vs. {X_train.shape[0]} in training set, {X_test.shape[1]} Features.")

# 6. Fitting and testing the model

# Import SVC from sklearn.svm and accuracy_score from sklearn.metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create an instance of the Support Vector Classification class
svc = SVC()

# Fit the model to the training data
svc.fit(X_train, y_train)

# Calculate accuracy scores on both train and test data
accuracy_train = accuracy_score(y_train, svc.predict(X_train))
accuracy_test = accuracy_score(y_test, svc.predict(X_test))

print(f"{accuracy_test:.1%} accuracy on test set vs. {accuracy_train:.1%} on training set")

# 7. Accuracy after dimensionality reduction

# Assign just the 'neckcircumferencebase' column from ansur_df to X
X = ansur_df[['neckcircumferencebase']]

# Split the data, instantiate a classifier and fit the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
svc = SVC()
svc.fit(X_train, y_train)

# Calculate accuracy scores on both train and test data
accuracy_train = accuracy_score(y_train, svc.predict(X_train))
accuracy_test = accuracy_score(y_test, svc.predict(X_test))

print(f"{accuracy_test:.1%} accuracy on test set vs. {accuracy_train:.1%} on training set")

# 8. Finding a good variance threshold

# Load the dataset
head_df = pd.read_csv('../../data/dimensionality_reduction_sources/ansur_head_df.csv')

# Create the boxplot
head_df.boxplot()

plt.show()

# Normalize the data
normalized_df = head_df / head_df.mean()

normalized_df.boxplot()
plt.show()

# Print the variances of the normalized data
print(normalized_df.var())

# 9. Features with low variance

from sklearn.feature_selection import VarianceThreshold

# Create a VarianceThreshold feature selector
sel = VarianceThreshold(threshold=0.001)

# Fit the selector to normalized head_df
sel.fit(normalized_df)

# Create a boolean mask
mask = sel.get_support()

# Apply the mask to create a reduced dataframe
reduced_df = head_df.loc[:, mask]

print(f"Dimensionality reduced from {normalized_df.shape[1]} to {reduced_df.shape[1]}.")

# 10. Removing features with many missing values

# Load the dataset
school_df = pd.read_csv('../../data/dimensionality_reduction_sources/boston.csv')

# Find the highest ratio of missing values for a single feature in the dataset
missing_ratios = school_df.isna().sum() / len(school_df)
mask = missing_ratios < 0.5

max_missing_ratio = missing_ratios.max()
print(f"Highest ratio of missing values for a single feature: {max_missing_ratio:.2%}")

# Create a boolean mask on whether each feature has less than 50% missing values
mask = school_df.columns[missing_ratios < 0.5]

# Create a reduced dataset by applying the mask
reduced_df = school_df.loc[:, mask]

print(school_df.shape)
print(reduced_df.shape)
print(f"Dimensionality reduced from {school_df.shape[1]} to {reduced_df.shape[1]}.")

# 11. Correlation and scale
ansur_df = pd.read_csv('../../data/dimensionality_reduction_sources/ansur_correlation.csv')
# Inspecting the correlation matrix
print(ansur_df.corr())

# Create the correlation matrix
corr = ansur_df.corr()
cmap = sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr,  cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
print(mask)

# Add the mask to the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()

# 11. Removing highly correlated features

ansur_df = pd.read_csv('../../data/dimensionality_reduction_sources/ANSUR_II_MALE.csv')
ansur_df = ansur_df.apply(pd.to_numeric, errors='coerce')

# Calculate the correlation matrix and take the absolute value
corr_df = ansur_df.corr().abs()

# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_df, dtype=bool))
tri_df = corr_df.mask(mask)

# List column names of highly correlated features (r > 0.95)
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]

# Drop the features in the to_drop list
reduced_df = ansur_df.drop(to_drop, axis=1)

print(f"The reduced dataframe has {reduced_df.shape[1]} columns.")

# 12. Nuclear energy and pool drownings

# Load the dataset
weird_df = pd.read_csv('../../data/dimensionality_reduction_sources/weird_df.csv')

# Put nuclear energy production on the x-axis and the number of pool drownings on the y-axis
sns.scatterplot(x='nuclear_energy', y='pool_drownings', data=weird_df)

# Show the plot
plt.show()

# Print out the correlation matrix of weird_df
print("Correlation between nuclear energy production and pool drownings:", weird_df.corr().loc['nuclear_energy', 'pool_drownings'])

# 13. Selecting features for model performance - Building a diabetes classifier

# Load the dataset
diabetes_df = pd.read_csv('../../data/dimensionality_reduction_sources/PimaIndians.csv')

X = diabetes_df.drop('test', axis=1)

y = diabetes_df['test']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
# Fit the scaler on the training features and transform these in one go
X_train_std = scaler.fit_transform(X_train)

# Fit the logistic regression model on the scaled training data
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_std, y_train)

# Scale the test features
X_test_std = scaler.transform(X_test)

# Predict diabetes presence on the scaled test set
y_pred = lr.predict(X_test_std)

# Prints accuracy metrics and feature coefficients
print(f"{accuracy_score(y_test, y_pred):.1%} accuracy on test set.")
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# 14. Manual Recursive Feature Elimination

from sklearn.feature_selection import RFE

# Remove the feature with the lowest model coefficient
X = diabetes_df[['pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'family', 'age']]

# Perform a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print("{0:.1%} accuracy on test set.".format(acc))
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# Remove the feature with the lowest model coefficient (in this case it's 'diastolic')
X = diabetes_df[['pregnant', 'glucose', 'triceps', 'insulin', 'bmi', 'family', 'age']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print(f"{acc:.1%} accuracy on test set. after removing diastolic")
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# Remove two more features with the lowest model coefficients
# Only keep the feature with the highest coefficient
X = diabetes_df[['glucose', 'triceps', 'bmi', 'family', 'age']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model to the data
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print(f"{acc:.1%} accuracy on test set. and removed 'insulin' and 'pregnant'")
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# Only keep the feature with the highest coefficient
X = diabetes_df[['glucose']]
# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model to the data
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print(f"{acc:.1%} accuracy on test set. and removed 'triceps', 'bmi', 'family', and 'age'")
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# 15. Automatic Recursive Feature Elimination
X = diabetes_df[['glucose', 'bmi', 'age']]
y = diabetes_df['test']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


rfe = RFE(estimator=LogisticRegression(), n_features_to_select=3, verbose=1)

# Fits the RFE selector to the training data
rfe.fit(scaler.fit_transform(X_train), y_train)

# Prints the features and their ranking (high = dropped early on)
print(dict(zip(X.columns, rfe.ranking_)))

# Print the features that are not eliminated
print(X.columns[rfe.support_])

# Calculates the test set accuracy
acc = accuracy_score(y_test, rfe.estimator_.predict(scaler.transform(X_test)))
print("{0:.1%} accuracy on test set.".format(acc))
