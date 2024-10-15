import ssl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ssl._create_default_https_context = ssl._create_unverified_context

# Many thanks to https://github.com/FraManl for his detailed commentary on this topic.

# Feature Extraction

# 1. Manual feature extraction 1
print("Manual feature extraction 1")

# load data
sales_df = pd.read_csv('../../data/dimensionality_reduction_sources/grocery_sales.csv')
print(sales_df.head())

# Calculate the price from the quantity sold and revenue
sales_df['price'] = sales_df['revenue'] / sales_df['quantity']

# Drop the quantity and revenue features
reduced_df = sales_df.drop(['quantity', 'revenue'], axis=1)

print(reduced_df.head())

# 2. Manual feature extraction 2
print("Manual feature extraction 2")

# Calculate the mean height
height_df = pd.read_csv('../../data/dimensionality_reduction_sources/ansur_height_df.csv')
print(height_df.head())

# Calculate the mean height
height_df['height'] = height_df[['height_1', 'height_2', 'height_3']].mean(axis=1)

# Drop the 3 original height features
reduced_df = height_df.drop(['height_1', 'height_2', 'height_3'], axis=1)

print(reduced_df.head())

# 3. Calculating Principal Components
print("Calculating Principal Components")

ansur_df_1 = pd.read_csv('../../data/dimensionality_reduction_sources/ANSUR_II_Male.csv')
ansur_df_2 = pd.read_csv('../../data/dimensionality_reduction_sources/ANSUR_II_Female.csv')
frames = (ansur_df_1, ansur_df_2)
ansur_df = pd.concat(frames)

non_numeric = ['Branch', 'Gender', 'Component', 'BMI_class', 'Height_class']
working_cls = ['headcircumference', 'buttockheight', 'waistcircumference', 'shouldercircumference']
ansur_df = ansur_df.drop(non_numeric, axis=1)
ansur_df = ansur_df.loc[:, working_cls]

# Create a pairplot to inspect ansur_df
sns.pairplot(data=ansur_df)
plt.show()

# Create the scaler
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)
# Create the PCA instance and fit and transform the data with pca
pca = PCA()
pc = pca.fit_transform(ansur_std)

# This changes the numpy array output back to a DataFrame
pc_df = pd.DataFrame(pc, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])

# Create a pairplot of the principal component dataframe
sns.pairplot(data=pc_df)
plt.show()

# 4. PCA on a larger dataset
print("PCA on a larger dataset")

# Enlarge dataset
ansur_df = pd.read_csv('../../data/dimensionality_reduction_sources/ANSUR_II_Female.csv')

non_numeric = ['Branch', 'Gender', 'Component', 'BMI_class', 'Height_class']
ansur_df = ansur_df.drop(non_numeric, axis=1)

col_vector = ansur_df.columns
picker = np.random.choice(col_vector, 13)
ansur_df = ansur_df.loc[:, picker]
print(ansur_df.head())

# Scale the data
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# 5. Apply PCA
pca = PCA()
pca.fit(ansur_std)

"""
You've fitted PCA on our 13 feature datasample. Now let's see how the components explain the variance.
"""

# Inspect the explained variance ratio per component
print("pca.explained_variance_ratio_:", pca.explained_variance_ratio_)

# Print the cumulative sum of the explained variance ratio
print("pca.explained_variance_ratio_.cumsum():", pca.explained_variance_ratio_.cumsum())

"""
What's the lowest number of principal components you should keep if you don't want to lose more than 10%
of explained variance during dimensionality reduction?
Using just 4 principal components we can explain more than 90% of the variance in the 13 feature dataset.
"""

# 6. Understanding the components
print("Understanding the components")

poke_df = pd.read_csv('../../data/dimensionality_reduction_sources/pokemon.csv')

poke_df = poke_df.drop(['Name', 'Type 1', 'Type 2'], axis=1)
print(poke_df.head())

# Build the pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(n_components=2))])

# Fit it to the dataset and extract the component vectors
pipe.fit(poke_df)
vectors = pipe.steps[1][1].components_.round(2)

# Print feature effects
print('PC 1 effects = ' + str(dict(zip(poke_df.columns, vectors[0]))))
print('PC 2 effects = ' + str(dict(zip(poke_df.columns, vectors[1]))))

# 7. PCA for feature exploration
print("PCA for feature exploration")
poke_df = pd.read_csv('../../data/dimensionality_reduction_sources/pokemon.csv')

poke_df = poke_df.drop(['Name', 'Type 1', 'Type 2'], axis=1)

# Build the pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                 ('reducer', PCA(n_components=2))])

# Fit the pipeline to poke_df and transform the data
pc = pipe.fit_transform(poke_df)
print("Principal Components:", pc)

poke_df = pd.read_csv('../../data/dimensionality_reduction_sources/pokemon.csv')
poke_df = poke_df.drop(['Name', 'Type 1', 'Type 2', 'Legendary'], axis=1)

poke2 = pd.read_csv('../../data/dimensionality_reduction_sources/pokemon.csv')
poke2 = poke2.drop(['Name', 'Type 2'], axis=1)

sel_col = ['Type 1', 'Legendary']
poke_cat_df = poke2.loc[:, sel_col]

pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(n_components=2))])

# Fit the pipeline to poke_df and transform the data
pc = pipe.fit_transform(poke_df)

# Add the 2 components to poke_cat_df
poke_cat_df['PC 1'] = pc[:, 0]
poke_cat_df['PC 2'] = pc[:, 1]
print(poke_cat_df.head())

# Use the Type feature to color the PC 1 vs PC 2 scatterplot
sns.scatterplot(data=poke_cat_df, x='PC 1', y='PC 2', hue='Type 1')
plt.show()

sns.scatterplot(data=poke_cat_df, x='PC 1', y='PC 2', hue='Legendary')
plt.show()

# 8. PCA in a model pipeline
poke_df = pd.read_csv('../../data/dimensionality_reduction_sources/pokemon.csv')
poke_df = poke_df.drop(['Name', 'Type 1', 'Type 2', 'Legendary'], axis=1)

poke2 = pd.read_csv('../../data/dimensionality_reduction_sources/pokemon.csv')
poke2 = poke2.drop(['Name', 'Type 2'], axis=1)
sel_col = ['Legendary']
poke_cat_df = poke2.loc[:, sel_col]

X = poke_df
y = poke_cat_df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Build the pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(n_components=3)),
                 ('classifier', RandomForestClassifier(random_state=0))])

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Score the accuracy on the test set
accuracy = pipe.score(X_test, y_test)

# Prints the explained variance ratio and accuracy
print(pipe.steps[1][1].explained_variance_ratio_)
print('{0:.1%} test set accuracy'.format(accuracy))

# 9. Selecting the proportion of variance to keep

ansur_df = pd.read_csv('../../data/dimensionality_reduction_sources/ANSUR_II_MALE.csv')

exclude = ['Branch', 'Component', 'Gender', 'BMI_class', 'Height_class']
ansur_df = ansur_df.drop(exclude, axis=1)

# Pipe a scaler to PCA selecting 80% of the variance
pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(n_components=0.8))])

# Fit the pipe to the data
pipe.fit(ansur_df)

print('{} components selected'.format(len(pipe.steps[1][1].components_)))

# Let PCA select 90% of the variance
pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(n_components=0.9))])

# Fit the pipe to the data
pipe.fit(ansur_df)

print('{} components selected'.format(len(pipe.steps[1][1].components_)))

# 10. Choosing the number of components
# Pipeline a scaler and pca selecting 10 components
pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(n_components=10))])

# Fit the pipe to the data
pipe.fit(ansur_df)

# Plot the explained variance ratio
plt.plot(pipe.steps[1][1].explained_variance_ratio_)

plt.xlabel('Principal component index')
plt.ylabel('Explained variance ratio')
plt.show()

# 11. PCA for image compression

# Load the data
(X_train, _), (X_test, _) = mnist.load_data()

# Reshape the data
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

def plot_digits(data):
    """Plot 8x8 digits in a 10x10 grid."""

    fig, axes = plt.subplots(4, 4, figsize=(10, 4),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(28, 28),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))

    plt.show()

plot_digits(X_test)

# Create the pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(n_components=78)), ('reconstructor', PCA(n_components=78))])

# Fit and transform the data
pipe.fit(X_train)

# Transform the input data to principal components
pc = pipe.transform(X_test)

# Prints the number of features per dataset
print(f"X_test has {X_test.shape[1]} features")
print(f"pc has {pc.shape[1]} features")

# Inverse transform the components to original feature space
X_rebuilt = pipe.inverse_transform(pc)

# Prints the number of features
print(f"X_rebuilt has {X_rebuilt.shape[1]} features")

# Plot the reconstructed data
plot_digits(X_rebuilt)