import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Removing features without variance

print("Removing features without variance")

# Load pokemon dataset
pokemon_df = pd.read_csv('../../data/8.dimensionality_reduction_sources/pokemon_gen1.csv')
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

print("Visually detecting redundant features")

# Load the dataset
ansur_df_1 = pd.read_csv('../../data/8.dimensionality_reduction_sources/ansur_df_1.csv')
ansur_df_2 = pd.read_csv('../../data/8.dimensionality_reduction_sources/ansur_df_2.csv')

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
ansur_df = pd.read_csv('../../data/8.dimensionality_reduction_sources/ansur_t-SNE.csv')

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

print("t-SNE visualisation of dimensionality")

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
print("The curse of dimensionality")
# 5. Train-test split

# Load the dataset
ansur_df = pd.read_csv('../../data/8.dimensionality_reduction_sources/ansur_test_train.csv')

# Select the Gender column as the feature to be predicted (y)
y = ansur_df['Gender']

# Remove the Gender column to create the training data
X = ansur_df.drop('Gender', axis=1)

# Perform a 70% train and 30% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(f"{X_test.shape[0]} rows in test set vs. {X_train.shape[0]} in training set, {X_test.shape[1]} Features.")

# 6. Fitting and testing the model
print("Fitting and testing the model")

# Create an instance of the Support Vector Classification class
svc = SVC()

# Fit the model to the training data
svc.fit(X_train, y_train)

# Calculate accuracy scores on both train and test data
accuracy_train = accuracy_score(y_train, svc.predict(X_train))
accuracy_test = accuracy_score(y_test, svc.predict(X_test))

print(f"{accuracy_test:.1%} accuracy on test set vs. {accuracy_train:.1%} on training set")

# 7. Accuracy after dimensionality reduction
print("Accuracy after dimensionality reduction")
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
print("Finding a good variance threshold")
# Load the dataset
head_df = pd.read_csv('../../data/8.dimensionality_reduction_sources/ansur_head_df.csv')

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
print("Features with low variance")

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
print("Removing features with many missing values")
# Load the dataset
school_df = pd.read_csv('../../data/8.dimensionality_reduction_sources/boston.csv')

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
print("Correlation and scale")
ansur_df = pd.read_csv('../../data/8.dimensionality_reduction_sources/ansur_correlation.csv')
# Inspecting the correlation matrix
print(ansur_df.corr())

# Create the correlation matrix
corr = ansur_df.corr()
cmap = sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr, cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
print(mask)

# Add the mask to the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()

# 11. Removing highly correlated features
print("Removing highly correlated features")

ansur_df = pd.read_csv('../../data/8.dimensionality_reduction_sources/ANSUR_II_MALE.csv')
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
print("Nuclear energy and pool drownings")
# Load the dataset
weird_df = pd.read_csv('../../data/8.dimensionality_reduction_sources/weird_df.csv')

# Put nuclear energy production on the x-axis and the number of pool drownings on the y-axis
sns.scatterplot(x='nuclear_energy', y='pool_drownings', data=weird_df)

# Show the plot
plt.show()

# Print out the correlation matrix of weird_df
print("Correlation between nuclear energy production and pool drownings:",
      weird_df.corr().loc['nuclear_energy', 'pool_drownings'])

# 13. Selecting features for model performance - Building a diabetes classifier
print("Selecting features for model performance - Building a diabetes classifier")
# Load the dataset
diabetes_df = pd.read_csv('../../data/8.dimensionality_reduction_sources/PimaIndians.csv')

predictors_vars = ['pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'family', 'age']
target_var = ['test']

X = diabetes_df[predictors_vars]
y = diabetes_df[target_var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
lr = LogisticRegression()

# Fit the scaler on the training features and transform these in one go
X_train_std = scaler.fit_transform(X_train)

# Fit the logistic regression model on the scaled training data
lr.fit(X_train_std, y_train)

# Scale the test features
X_test_std = scaler.transform(X_test)

# Predict diabetes presence on the scaled test set
y_pred = lr.predict(X_test_std)

# Prints accuracy metrics and feature coefficients
print("{0:.1%} accuracy on test set.".format(accuracy_score(y_test, y_pred)))
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# 14. Manual Recursive Feature Elimination
print("Manual Recursive Feature Elimination")

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
print("Automatic Recursive Feature Elimination")
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

# 16. Building a random forest model
print("Building a random forest model")

X = diabetes_df.drop('test', axis=1)

y = diabetes_df['test']

# Perform a 75% training and 25% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fit the random forest model to the training data
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Calculate the test set accuracy
acc = accuracy_score(y_test, rf.predict(X_test))
print(f"{acc:.1%} accuracy on test set.")

# Print the importance per feature
print("Print the importance per feature:")
print(dict(zip(X.columns, rf.feature_importances_.round(2))))

# Print accuracy
print("{0:.1%} accuracy on test set.".format(acc))

# 17. Random forest for feature selection
print("Random forest for feature selection")
# Fit the random forest model to the training data
rf.fit(X_train, y_train)

# Create a mask for features with an importance higher than 0.15
mask = rf.feature_importances_ > 0.15

# Prints out the mask
print(mask)

# Apply the mask to the feature dataset X
reduced_X = X.loc[:, mask]

# prints out the selected column names
print(reduced_X.columns)

# Recursive Feature Elimination with random forests

# Create the RFE with a RandomForestClassifier estimator and 3 features to select
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=2, verbose=1)

# Fit the eliminator to the data
rfe.fit(X_train, y_train)

# Print the features and their ranking (high = dropped early on)
print(dict(zip(X.columns, rfe.ranking_)))

# Print the features that are not eliminated
print(X.columns[rfe.support_])

# Create a mask for using the support_ attribute of rfe
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print("Applying the mask to feature dataset X and printing the result", reduced_X.columns)

# 18. Random forest accuracy
print("Random forest accuracy")
# Transform X with RFE selector
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Fit the random forest model to the training data
rf.fit(X_train_rfe, y_train)

# Calculate the test set accuracy
acc = accuracy_score(y_test, rf.predict(X_test_rfe))
print("{0:.1%} accuracy on test set.".format(acc))

# Wrap the feature eliminator around the random forest model
rfe_rf = RFE(estimator=RandomForestClassifier(), n_features_to_select=3, verbose=1, step=2)

# Fit the combined model to the training data
rfe_rf.fit(X_train, y_train)

# Calculate the test set accuracy
acc = accuracy_score(y_test, rfe_rf.predict(X_test))
print("{0:.1%} accuracy on test set.".format(acc))

# FROM THIS POINT ON THE CODE IS NOT WORKING CORRECTLY. THIS IS DUE TO THE VARIANCES IN THE ANSUR DATASET


# 19. Creating a LASSO regressor
print("Creating a LASSO regressor")

ansur_male_df = pd.read_csv('../../data/8.dimensionality_reduction_sources/ANSUR_II_MALE.csv')
ansur_female_df = pd.read_csv('../../data/8.dimensionality_reduction_sources/ANSUR_II_FEMALE.csv')

frames = [ansur_male_df, ansur_female_df]
df = pd.concat(frames)

non_numeric = ['Branch', 'Gender', 'Component', 'BMI_class', 'Height_class']
# Select the Gender column as the feature to be predicted (y)
y = df['BMI']

# Remove the Gender column to create the training data
X = df.drop(non_numeric, axis=1)

# Set the test size to 30% to get a 70-30% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit the scaler on the training features and transform these in one go
X_train_std = scaler.fit_transform(X_train, y_train)

# Create the Lasso model
la = Lasso()

# Fit it to the standardized training data
la.fit(X_train_std, y_train)

# 20. Lasso model results
print("Lasso model results")

# Transform the test set with the pre-fitted scaler
X_test_std = scaler.transform(X_test)

# Calculate the coefficient of determination (R squared) on the scaled test set X_test_std
r_squared = la.score(X_test_std, y_test)
print(f"R squared: {r_squared:.2f}")
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))

# Create a list that has True values when coefficients equal 0
zero_coef = la.coef_ == 0

# Calculate how many features have a zero coefficient
n_ignored = sum(zero_coef)
print(f"The model has ignored {n_ignored} out of {len(la.coef_)} features.")

# 21. Adjusting the regularization strength
print("Adjusting the regularization strength")
# Find the highest alpha value with R-squared above 98%
la = Lasso(alpha=0.1, random_state=0)

# Fits the model and calculates performance stats
la.fit(X_train_std, y_train)
r_squared = la.score(X_test_std, y_test)
n_ignored_features = sum(la.coef_ == 0)

# Print performance stats
print(f"The model can predict {r_squared:.1%} of the variance in the test set.")
print(f"The model has ignored {n_ignored_features} out of {len(la.coef_)} features.")

# Creating a LassoCV regressor

# Create and fit the LassoCV model on the training set
lcv = LassoCV()
lcv.fit(X_train, y_train)
print('Optimal alpha = {0:.3f}'.format(lcv.alpha_))

# Calculate R squared on the test set
r_squared = lcv.score(X_test, y_test)
print('The model explains {0:.1%} of the test set variance'.format(r_squared))

# Create a mask for coefficients not equal to zero
lcv_mask = lcv.coef_ != 0
print('{} features out of {} selected'.format(sum(lcv_mask), len(lcv_mask)))

# 22. Ensemble models for extra votes
print("Ensemble models for extra votes")

# Select 10 features with RFE on a GradientBoostingRegressor, drop 3 features on each step
rfe_gb = RFE(estimator=GradientBoostingRegressor(), n_features_to_select=10, step=3, verbose=1)
rfe_gb.fit(X_train, y_train)

# Calculate the R squared on the test set
r_squared = rfe_gb.score(X_test, y_test)
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))

# Assign the support array to gb_mask
gb_mask = rfe_gb.support_

# Select 10 features with RFE on a RandomForestRegressor, drop 3 features on each step
rfe_rf = RFE(estimator=RandomForestRegressor(), n_features_to_select=10, step=3, verbose=1)
rfe_rf.fit(X_train, y_train)

# Calculate the R squared on the test set
r_squared = rfe_rf.score(X_test, y_test)
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))

# Assign the support array to gb_mask
rf_mask = rfe_rf.support_

# Combining 3 feature selectors
# Sum the votes of the three models
votes = np.sum([lcv_mask, gb_mask, rf_mask], axis=0)
print(votes)

# Create a mask for features selected by all 3 models
meta_mask = votes >= 3
print(meta_mask)

# Apply the dimensionality reduction on X
X_reduced = X.loc[:, meta_mask]
print(X_reduced.columns)

# Plug the reduced dataset into a linear regression pipeline

lm = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=0)
lm.fit(scaler.fit_transform(X_train), y_train)
r_squared = lm.score(scaler.transform(X_test), y_test)
print('The model can explain {0:.1%} of the variance in the test set using {1} features.'.format(r_squared,
                                                                                                 len(lm.coef_)))
