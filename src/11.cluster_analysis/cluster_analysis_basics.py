# Import pandas library
import pandas as pd
# Import the image class of the matplotlib library
import matplotlib.pyplot as plt
# Import seaborn library
import seaborn as sns
# import image class of matplotlib (for Batman image)
from matplotlib import image as img
# Import the timeit module
import timeit
# Import the random class
from numpy import random
# Import the dendrogram function
from scipy.cluster.hierarchy import dendrogram
# Import the fcluster and linkage functions
from scipy.cluster.hierarchy import fcluster, linkage
# Import the kmeans and vq functions
from scipy.cluster.vq import kmeans, vq
# Import the whiten function
from scipy.cluster.vq import whiten
## import nltk (for remove_noise function)
# import nltk
## import re (for remove_noise function)
# import re
## Import the word_tokenize function for the remove_noise function
# from nltk.tokenize import word_tokenize
## Import TfidfVectorizer class from sklearn (for documents)
# from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Pokemon Sightings
print("-----------------Pokemon Sightings-----------------")
x = [9, 6, 2, 3, 1, 7, 1, 6, 1, 7, 23, 26, 25, 23, 21, 23, 23, 20, 30, 23]
y = [8, 4, 10, 6, 0, 4, 10, 10, 6, 1, 29, 25, 30, 29, 29, 30, 25, 27, 26, 30]

# Create a scatter plot
plt.scatter(x, y)

# Display the scatter plot
plt.show()
# --------------------------------------------------------------------------
# Pokémon Sightings: Hierarchical clustering
print("-----------------Pokémon Sightings: Hierarchical clustering-----------------")

pokemon_df = pd.DataFrame({'x': x, 'y': y})

# Use the linkage() function
Z = linkage(pokemon_df, 'ward')

# Assign cluster labels
pokemon_df['cluster_labels'] = fcluster(Z, 2, criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=pokemon_df)

plt.show()
# --------------------------------------------------------------------------
# Pokémon Sightings: K-Means Clustering
print("-----------------Pokémon Sightings: K-Means Clustering-----------------")

# Compute cluster centers after typecasting integers as float
centroids, _ = kmeans(pokemon_df[['x', 'y']].values.astype(float), 2)

# Assign cluster labels
pokemon_df['cluster_labels'], _ = vq(pokemon_df[['x', 'y']].values.astype(float), centroids)

# Plot clusters
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=pokemon_df)
plt.show()
# --------------------------------------------------------------------------
# Normalize basic list data
print("-----------------2. Normalize basic list data-----------------")

goals_for = [4, 3, 2, 3, 1, 1, 2, 0, 1, 4]

# Use the whiten() function to standardize the data
scaled_data = whiten(goals_for)
print(scaled_data)
# --------------------------------------------------------------------------
# Visualize normalized data
print("-----------------Visualize normalized data-----------------")

# Plot original data
plt.plot(goals_for, label='original')

# Plot scaled data
plt.plot(scaled_data, label='scaled', linestyle='--')

# Show the legend in the plot
plt.legend()

# Display the plot
plt.show()
# --------------------------------------------------------------------------
# Normalization of small numbers
print("-----------------Normalization of small numbers-----------------")

# Prepare data
rate_cuts = [0.0025, 0.001, -0.0005, -0.001, -0.0005, 0.0025, -0.001, -0.0015, -0.001, 0.0005]

# Use the whiten() function
scaled_data = whiten(rate_cuts)

# Plot original data
plt.plot(rate_cuts, label='original')

# Plot scaled data
plt.plot(scaled_data, label='scaled', linestyle='--')

# Show the legend in the plot
plt.legend()

# Display the plot
plt.show()
# --------------------------------------------------------------------------
# FIFA 18: Normalize data
print("-----------------FIFA 18: Normalize data-----------------")

# load the data
fifa = pd.read_csv('../../data/11.cluster_analysis_sources/fifa_18_sample_data.csv')

# Scale wage and value
fifa['scaled_wage'] = whiten(fifa['eur_wage'])
fifa['scaled_value'] = whiten(fifa['eur_value'])

# Plot the two columns in a scatter plot
fifa.plot(x='scaled_wage', y='scaled_value', kind='scatter')
plt.show()

# Check the mean and standard deviation of the scaled data
print(fifa[['scaled_wage', 'scaled_value']].describe())
# --------------------------------------------------------------------------

# 2. Basics of Hierarchical clustering
print("-----------------2. Basics of Hierarchical clustering-----------------")

# Hierarchical clustering: ward method
print("-----------------Hierarchical clustering: ward method-----------------")

# Load the data
comic_con = pd.read_csv('../../data/11.cluster_analysis_sources/comic_con.csv')

# Use the linkage() function
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method='ward', metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', hue='cluster_labels', data=comic_con)
plt.title('Comic Con Cluster using ward method')
plt.show()
# --------------------------------------------------------------------------

# Hierarchical clustering: single method
print("-----------------Hierarchical clustering: single method-----------------")

# Use the linkage() function
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method='single', metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', hue='cluster_labels', data=comic_con)
plt.title('Comic Con Cluster using single method')
plt.show()
# --------------------------------------------------------------------------

# Hierarchical clustering: complete method
print("-----------------Hierarchical clustering: complete method-----------------")

# Use the linkage() function
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method='complete', metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', hue='cluster_labels', data=comic_con)
plt.title('Comic Con Cluster using complete method')
plt.show()
# --------------------------------------------------------------------------
# Visualize clusters with matplotlib
print("-----------------Visualize clusters with matplotlib-----------------")

# Define a colors dictionary for clusters
colors = {1: 'red', 2: 'blue'}

# Plot a scatter plot
comic_con.plot.scatter(x='x_scaled', y='y_scaled', c=comic_con['cluster_labels'].apply(lambda x: colors[x]))
plt.title('Comic Con Cluster - visualizing with matplotlib')
plt.show()
# --------------------------------------------------------------------------
# Visualize clusters with seaborn
print("-----------------Visualize clusters with seaborn-----------------")

# Plot a scatter plot using seaborn
sns.scatterplot(x='x_scaled', y='y_scaled', hue='cluster_labels', data=comic_con)
plt.title('Comic Con Cluster - visualizing with seaborn')
plt.show()
# --------------------------------------------------------------------------
# Create a dendrogram
print("-----------------Create a dendrogram-----------------")

# Create a dendrogram
dn = dendrogram(distance_matrix)

# Display the dendogram
plt.title('Comic Con Cluster - visualizing with Dendrogram')
plt.show()
# --------------------------------------------------------------------------
# Timing run of hierarchical clustering
print("-----------------Timing run of hierarchical clustering-----------------")

# Use the timeit magic command
print("Time taken to run hierarchical clustering 10 times using ward method:")
print(timeit.timeit('linkage(comic_con[["x_scaled", "y_scaled"]], method="ward", metric="euclidean")', number=10,
                    setup='from __main__ import linkage, comic_con'))
# --------------------------------------------------------------------------
# FIFA 18: exploring defenders
print("-----------------FIFA 18: exploring defenders-----------------")

# Load the data
fifa = pd.read_csv('../../data/11.cluster_analysis_sources/fifa_18_dataset_scaled.csv')

# Fit the data into a hierarchical clustering algorithm and time the run using the timeit magic command
print("Time taken to run hierarchical clustering 10 times using ward method:")
print(timeit.timeit('linkage(fifa[["scaled_sliding_tackle", "scaled_aggression"]], method="ward", metric="euclidean")',
                    number=10, setup='from __main__ import linkage, fifa'))

# Fit the data into a hierarchical clustering algorithm, using the ward method
distance_matrix = linkage(fifa[['scaled_sliding_tackle', 'scaled_aggression']], 'ward')

# Assign cluster labels to each row of data
fifa['cluster_labels'] = fcluster(distance_matrix, 3, criterion='maxclust')

# Display cluster centers of each cluster
print(fifa[['scaled_sliding_tackle', 'scaled_aggression', 'cluster_labels']].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='scaled_sliding_tackle', y='scaled_aggression', hue='cluster_labels', data=fifa)
plt.title('FIFA 18 - Defender Clusters')
plt.show()
# --------------------------------------------------------------------------
# 3. Basics of K-Means clustering
print("-----------------3. Basics of K-Means clustering-----------------")

# KMeans clustering: first exercise
print("-----------------KMeans clustering: first exercise-----------------")

# Import the kmeans and vq functions

# Generate cluster centers
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], 2)

# Assign cluster labels
comic_con['cluster_labels'], distortion_list = vq(comic_con[['x_scaled', 'y_scaled']], cluster_centers)

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', hue='cluster_labels', data=comic_con)
plt.title('Comic Con Cluster using KMeans')
plt.show()
# --------------------------------------------------------------------------
# Time KMeans clustering run
print("-----------------Time KMeans clustering run-----------------")

# Fit the data into a KMeans algorithm
print("Time taken to run KMeans clustering 10 times:")
print(timeit.timeit('kmeans(comic_con[["x_scaled", "y_scaled"]], 2)', number=10,
                    setup='from __main__ import kmeans, comic_con'))
# --------------------------------------------------------------------------

# KMeans clustering: elbow method
print("-----------------KMeans clustering: elbow method-----------------")

distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], i)
    distortions.append(distortion)

# Create a data frame with two lists - number of clusters and distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.title('KMeans - Elbow Plot')
plt.show()
# --------------------------------------------------------------------------
# Elbow method on uniform data
print("-----------------Elbow method on uniform data-----------------")

# Load the data
uniform_data = pd.read_csv('../../data/11.cluster_analysis_sources/k-means_uniform_data_scaled.csv')

# Fit the data into a KMeans algorithm
distortions = []
num_clusters = range(2, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(uniform_data[['x_scaled', 'y_scaled']], i)
    distortions.append(distortion)

# Create a data frame with two lists - number of clusters and distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.title('KMeans - Elbow Plot')
plt.show()
# --------------------------------------------------------------------------
# Impact of seeds on distinct clusters
print("-----------------Impact of seeds on distinct clusters-----------------")

# Initialize seed
random.seed([1, 2, 1000])

# Run kmeans clustering
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], 2)
comic_con['cluster_labels'], distortion_list = vq(comic_con[['x_scaled', 'y_scaled']], cluster_centers)

# Plot the scatterplot
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data=comic_con)
plt.title('Comic Con Cluster using KMeans and random seed')
plt.show()
# --------------------------------------------------------------------------
# Uniform clustering patterns
print("-----------------Uniform clustering patterns-----------------")

# Load the mouse-like dataset
mouse = pd.read_csv('../../data/11.cluster_analysis_sources/k-means_mouse.csv')

# Generate cluster centers from the mouse dataset
cluster_centers, distortion = kmeans(mouse[['x_scaled', 'y_scaled']], 3)

# Assign cluster labels to the mouse dataset
mouse['cluster_labels'], distortion_list = vq(mouse[['x_scaled', 'y_scaled']], cluster_centers)

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', hue='cluster_labels', data=mouse)
plt.title('Uniform data patterns with mouse-like dataset')
plt.show()
# --------------------------------------------------------------------------
# FIFA 18: defenders revisited
print("-----------------FIFA 18: defenders revisited-----------------")

# Import the data
fifa = pd.read_csv('../../data/11.cluster_analysis_sources/fifa_18_k-means_scaled.csv')

# Set up a random seed in numpy
random.seed([1000, 2000])

# Fit the data into a KMeans algorithm
cluster_centers, distortion = kmeans(fifa[['scaled_def', 'scaled_phy']], 3)

# Assign cluster labels
fifa['cluster_labels'], _ = vq(fifa[['scaled_def', 'scaled_phy']], cluster_centers)

# Display cluster centers of each cluster
print(fifa[['scaled_def', 'scaled_phy', 'cluster_labels']].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='scaled_def', y='scaled_phy', hue='cluster_labels', data=fifa)
plt.title('FIFA 18 - Defender Clusters using K-means')
plt.show()
# --------------------------------------------------------------------------

# 4. Clustering in Real World
print("-----------------4. Clustering in Real World-----------------")

# read batman image and print dimensions
batman_image = img.imread('../../data/11.cluster_analysis_sources/batman.jpg')
print("Batman image dimensions: ", batman_image.shape)

# Store RGB values of all pixels in lists r, g and b
r = []
g = []
b = []
for row in batman_image:
    for temp_r, temp_g, temp_b in row:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)

## This is alternate code - works but not needed for the course
# Create a DataFrame with the RGB values
batman_df = pd.DataFrame({'red': r,
                          'blue': b,
                          'green': g})

# Use the whiten() function to standardize the data
batman_df['scaled_red'] = whiten(batman_df['red'])
batman_df['scaled_blue'] = whiten(batman_df['blue'])
batman_df['scaled_green'] = whiten(batman_df['green'])

# Fit the data into a KMeans algorithm
cluster_centers, distortion = kmeans(batman_df[['scaled_red', 'scaled_blue', 'scaled_green']], 3)

# Assign cluster labels
batman_df['cluster_labels'], _ = vq(batman_df[['scaled_red', 'scaled_blue', 'scaled_green']], cluster_centers)

# Display cluster centers of each cluster
print(batman_df[['scaled_red', 'scaled_blue', 'scaled_green', 'cluster_labels']].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='scaled_red', y='scaled_blue', hue='cluster_labels', data=batman_df)
plt.title('Batman Image Clustered')
plt.show()

# --------------------------------------------------------------------------
distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(batman_df[['scaled_red', 'scaled_blue', 'scaled_green']], i)
    distortions.append(distortion)

# Create a data frame with two lists - number of clusters and distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Create a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.title('KMeans - Elbow Plot')
plt.show()
# --------------------------------------------------------------------------

# Get dominant colors
print("-----------------Get dominant colors-----------------")

# Get standard deviations of each color
r_std, g_std, b_std = batman_df[['red', 'green', 'blue']].std()

# Get dominant color
colors = []
for cluster_center in cluster_centers:
    scaled_r, scaled_g, scaled_b = cluster_center
    colors.append((
        scaled_r * r_std / 255,
        scaled_g * g_std / 255,
        scaled_b * b_std / 255
    ))

# Display colors of cluster centers
plt.imshow([colors])
plt.title('Batman Image - Cluster Center Colors')
plt.axis('off')
plt.show()
# --------------------------------------------------------------------------
# TD-IDF of movie plots
print("-----------------TD-IDF of movie plots-----------------")
# The below code does not work since nltk package installation is throwing errors.
# # download the punkt tokenizer for sentence splitting
# nltk.download('punkt')
# nltk.download('punkt_tab')
#
#
# # Define a tokenizer and remove noise
# def remove_noise(text, stop_words=[]):
#     tokens = word_tokenize(text)
#     cleaned_tokens = []
#     for token in tokens:
#         token = re.sub('[^A-Za-z0-9]+', '', token)
#         if len(token) > 1 and token.lower() not in stop_words:
#             # Get lowercase
#             cleaned_tokens.append(token.lower())
#     return cleaned_tokens
#
#
# # Load the data
# movies = pd.read_csv('../../data/11.cluster_analysis_sources/movies_plot.csv')
#
# # Create a list of plots
# plots = movies['Plot']
#
# # Initialize TfidfVectorizer
# tfidf_vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.75, max_features=50, tokenizer=remove_noise)
#
# # Use the .fit_transform() method on the list plots
# tfidf_matrix = tfidf_vectorizer.fit_transform(plots)
#
# # Top terms in each cluster
# print("-----------------Top terms in each cluster-----------------")

# num_clusters = 2
#
# # Generate cluster centers through the kmeans function
# cluster_centers, distortion = kmeans(tfidf_matrix.todense(), num_clusters)
#
# # Generate terms from the tfidf_vectorizer object
# terms = tfidf_vectorizer.get_feature_names_out()
#
# # Sort the terms and print top terms per cluster
# for i in range(num_clusters):
#     center_terms = dict(zip(terms, list(cluster_centers[i])))
#     sorted_terms = sorted(center_terms, key=center_terms.get, reverse=True)
#     print("Cluster ", i, ": ", sorted_terms[:5])

# --------------------------------------------------------------------------
# Basic checks on clusters
print("-----------------Basic checks on clusters-----------------")

# Load the data
fifa = pd.read_csv('../../data/11.cluster_analysis_sources/fifa_18_k-means_scaled.csv')

fifa['scaled_pac'] = whiten(fifa['pac'])
fifa['scaled_dri'] = whiten(fifa['dri'])
fifa['scaled_sho'] = whiten(fifa['sho'])

# Fit the data into a KMeans algorithm
cluster_centers, distortion = kmeans(fifa[['scaled_dri', 'scaled_sho', 'scaled_pac']], 3)

# Assign cluster labels
fifa['cluster_labels'], _ = vq(fifa[['scaled_dri', 'scaled_sho', 'scaled_pac']], cluster_centers)

# Display cluster centers of each cluster
print(fifa[['scaled_dri', 'scaled_sho', 'scaled_pac', 'cluster_labels']].groupby('cluster_labels').mean())

# Print the size of the clusters
print(fifa.groupby('cluster_labels')['ID'].count())

# Print the mean value of wages in each cluster
print(fifa.groupby('cluster_labels')['eur_wage'].mean())
# --------------------------------------------------------------------------
# FIFA 18: what makes a complete player?
print("-----------------FIFA 18: what makes a complete player?-----------------")
fifa['scaled_pac'] = whiten(fifa['pac'])
fifa['scaled_dri'] = whiten(fifa['dri'])
fifa['scaled_sho'] = whiten(fifa['sho'])
fifa['scaled_pas'] = whiten(fifa['pas'])
fifa['scaled_def'] = whiten(fifa['def'])
fifa['scaled_phy'] = whiten(fifa['phy'])

scaled_features = ['scaled_pac', 'scaled_dri', 'scaled_sho', 'scaled_pas', 'scaled_def', 'scaled_phy']

# Fit the data into a KMeans algorithm
cluster_centers, _ = kmeans(fifa[scaled_features], 2)

# Assign cluster labels and print cluster centers
fifa['cluster_labels'], _ = vq(fifa[scaled_features], cluster_centers)
print(fifa.groupby('cluster_labels')[scaled_features].mean())

# Plot cluster centers to visualize clusters
fifa.groupby('cluster_labels')[scaled_features].mean().plot(legend=True, kind='bar')
plt.title('FIFA 18 Attributes by Cluster')
plt.show()

# Get the names column of first 5 players in each cluster
for cluster in fifa['cluster_labels'].unique():
    print(fifa[fifa['cluster_labels'] == cluster]['name'].values[:5])
# --------------------------------------------------------------------------
