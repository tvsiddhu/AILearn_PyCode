# 1. Plotting a histogram of iris data

# Import plotting modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()

# Convert to a pandas DataFrame for easier manipulation
# The iris dataset has features for sepal length, sepal width, petal length, and petal width
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target species (0: Setosa, 1: Versicolor, 2: Virginica) to the DataFrame
iris_df['species'] = iris.target

# Filter for rows where the species is 1 (which corresponds to Versicolor)
versicolor_df = iris_df[iris_df['species'] == 1]

# Extract the petal length (third column) for Versicolor
# Petal length corresponds to the third feature (index 2 in 0-based indexing)
versicolor_petal_length = versicolor_df['petal length (cm)']
versicolor_petal_width = versicolor_df['petal width (cm)']

# Filter for rows where the species is 0 (which corresponds to Setosa)
setosa_df = iris_df[iris_df['species'] == 0]
setosa_petal_length = setosa_df['petal length (cm)']
setosa_petal_width = setosa_df['petal width (cm)']
# Filter for rows where the species is 2 (which corresponds to Virginica)
virginica_df = iris_df[iris_df['species'] == 2]
virginica_petal_length = virginica_df['petal length (cm)']
virginica_petal_width = virginica_df['petal width (cm)']
# Set default Seaborn style
sns.set_theme()
sns.color_palette("Spectral", as_cmap=True)

# Plot histogram of versicolor petal lengths
plt.hist(versicolor_petal_length)

# Show histogram
plt.show()

# 2. Axis labels!
# Plot histogram of versicolor petal lengths
plt.hist(versicolor_petal_length)

# Label axes
plt.title('Histogram of Versicolor Petal Lengths')
plt.xlabel('petal length (cm)')
plt.ylabel('count')

# Show histogram
plt.show()

# 3. Adjusting the number of bins in a histogram

# Compute number of data points: n_data
n_data = len(versicolor_petal_length)

# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)

# Convert number of bins to integer: n_bins
n_bins = int(n_bins)

# Plot histogram of versicolor petal lengths
plt.hist(versicolor_petal_length, bins=n_bins)

# Label axes
plt.title('Histogram of Versicolor Petal Lengths')
plt.xlabel('petal length (cm)')
plt.ylabel('count')

# Show histogram
plt.show()

# 4. Bee swarm plot
# Create bee swarm plot with Seaborn's default settings
sns.swarmplot(x='species', y='petal length (cm)', data=iris_df)

# Label the axes
plt.title('Bee Swarm Plot')
plt.xlabel('species')
plt.ylabel('petal length (cm)')

# Show the plot
plt.show()


# 5. ECDFs or Empirical Cumulative Distribution Functions

# Define a function to compute the ECDF
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y


# Compute ECDF for versicolor data: x_vers, y_vers

x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
plt.plot(x_vers, y_vers, marker='.', linestyle='none')

# Label the axes
plt.title('ECDF of Versicolor Petal Lengths')
plt.xlabel('petal length (cm)')
plt.ylabel('ECDF')

# Display the plot
plt.show()

# 6. Comparison of ECDFs

# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

# Plot all ECDFs on the same plot
plt.plot(x_set, y_set, marker='.', linestyle='none')
plt.plot(x_vers, y_vers, marker='.', linestyle='none')
plt.plot(x_virg, y_virg, marker='.', linestyle='none')

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
plt.title('ECDF of Petal Lengths')
plt.xlabel('petal length (cm)')
plt.ylabel('ECDF')

# Display the plot
plt.show()

# 7. Computing Means

# Compute the mean: mean_length_vers
mean_length_vers = np.mean(versicolor_petal_length)

# Print the result with some nice formatting
print("******************\nMean Length of Versicolor Petal\n******************\n", mean_length_vers, 'cm')

# 8. Computing Percentiles

# Specify array of percentiles: percentiles
percentiles = np.array([2.5, 25, 50, 75, 97.5])

# Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length, percentiles)

# Print the result
print("******************\nComputer Percentiles Array\n******************\n", ptiles_vers)

# 9. Comparing percentiles to ECDF

# Plot the ECDF
plt.plot(x_vers, y_vers, '.')
plt.xlabel('petal length (cm)')
plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
plt.plot(ptiles_vers, percentiles / 100, marker='D', color='red', linestyle='none')

# Show the plot
plt.show()

# 10. Box-and-whisker plot

# Create box plot with Seaborn's default settings
sns.boxplot(x='species', y='petal length (cm)', data=iris_df)

# Label the axes
plt.title('Box plot')
plt.xlabel('species')
plt.ylabel('petal length (cm)')

# Show the plot
plt.show()

# 11. Computing the variance

# Array of differences to mean: differences
differences = versicolor_petal_length - np.mean(versicolor_petal_length)

# Square the differences: diff_sq
diff_sq = differences ** 2

# Compute the mean square difference: variance_explicit
variance_explicit = np.mean(diff_sq)

# Compute the variance using NumPy: variance_np
variance_np = np.var(versicolor_petal_length)

# Print the results
print("******************\nComputing the Variances\n******************\n")
print("Variance using the explicit formula: ", variance_explicit, "cm^2")
print("Variance using NumPy: ", variance_np, "cm^2")

# 12. The standard deviation and the variance

# Compute the variance: variance
variance = np.var(versicolor_petal_length)

# Print the square root of the variance
print("******************\nComputing the Standard Deviation\n******************\n")
print("Standard Deviation: ", np.sqrt(variance), "cm")

# Print the standard deviation
print("Standard Deviation: ", np.std(versicolor_petal_length), "cm")

# 13. Scatter plot

# Make a scatter plot
plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')

# Label the axes
plt.title('Versicolor Petal Length vs. Width')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')

# Show the result
plt.show()

# 14. Computing the covariance

# Compute the covariance matrix: covariance_matrix
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)

# Print covariance matrix
print("******************\nComputing the Covariance Matrix\n******************\n")
print(covariance_matrix)

# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0, 1]

# Print the result
print("Covariance of petal length and petal width: ", petal_cov)


# 15. Computing the Pearson correlation coefficient

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0, 1]


# Compute Pearson correlation coefficient for I. versicolor: r
r = pearson_r(versicolor_petal_length, versicolor_petal_width)

# Print the result
print("******************\nComputing the Pearson Correlation Coefficient\n******************\n")
print("Pearson correlation coefficient: ", r)

# 16. Plotting the covariance

# Plot the covariance matrix
sns.heatmap(covariance_matrix, annot=True)

# Label the axes
plt.title('Covariance Matrix')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')

# Show the plot
plt.show()

# 17. Plotting the Pearson correlation coefficient

# Plot the covariance matrix
sns.heatmap(covariance_matrix, annot=True)

# Label the axes
plt.title('Pearson Correlation Coefficient')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.show()

# 18. Generating random numbers using the np.random module

# Instantiate and seed the random number generator

rng = np.random.default_rng(42)

# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = rng.random()

# Plot a histogram
plt.hist(random_numbers)

# Show the plot
plt.show()


# 19. The np.random module and Bernoulli trials

def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0

    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = rng.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success


# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100, 0.05)

# Plot the histogram with default number of bins; label axes
plt.hist(n_defaults, density=True)
plt.title('Bernoulli Trials')
plt.xlabel('number of defaults out of 100 loans')
plt.ylabel('probability')

# Show the plot
plt.show()

# 20. Will the bank fail?

# Compute ECDF: x, y
x, y = ecdf(n_defaults)

# Plot the ECDF with labeled axes
plt.plot(x, y, marker='.', linestyle='none')
plt.title('ECDF of Number of Defaults')
plt.xlabel('number of defaults out of 100 loans')
plt.ylabel('ECDF')

# Show the plot
plt.show()

# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
n_lose_money = np.sum(n_defaults >= 10)

# Compute and print probability of losing money
print("Probability of losing money: ", n_lose_money / len(n_defaults))

# 21. Probability distributions and stories: The Binomial distribution

# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = rng.binomial(100, 0.05, size=10000)

# Compute CDF: x, y
x, y = ecdf(n_defaults)

# Plot the CDF with axis labels
plt.plot(x, y, marker='.', linestyle='none')
plt.title('CDF of Number of Defaults')
plt.xlabel('number of defaults out of 100 loans')
plt.ylabel('CDF')

# Show the plot
plt.show()

# 22. Plotting the Binomial PMF

# Compute bin edges: bins
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5

# Generate histogram
plt.hist(n_defaults, bins=bins, density=True)

# Label axes
plt.title('Binomial PMF')
plt.xlabel('number of defaults out of 100 loans')
plt.ylabel('PMF')

# Show the plot
plt.show()

# 23. Relationship between Binomial and Poisson distributions

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = rng.poisson(10, size=10000)

# Print the mean and standard deviation
print("******************\nRelationship between Binomial and Poisson distributions\n******************\n")

print("Poisson Mean:     ", np.mean(samples_poisson), "\nPoisson Standard Deviation: ", np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]

# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = rng.binomial(n[i], p[i], size=10000)

    # Print results
    print("n =", n[i], "Binomial Mean:", np.mean(samples_binomial), "Binomial Standard Deviation:",
          np.std(samples_binomial))

# 24. Was 2015 anomalous?

# Draw 100,000 samples out of Poisson distribution: n_nohitters
n_nohitters = rng.poisson(251 / 115, size=10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)

# Compute probability of getting seven or more: p_large
p_large = n_large / 10000

# Print the result
print("Probability of seven or more no-hitters:", p_large)

# 25. The Normal PDF (Probability Density Function)

# Draw 100,000 samples from a Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = rng.normal(20, 1, size=100000)
samples_std3 = rng.normal(20, 3, size=100000)
samples_std10 = rng.normal(20, 10, size=100000)

# Make histograms
plt.hist(samples_std1, bins=100, density=True, histtype='step')
plt.hist(samples_std3, bins=100, density=True, histtype='step')
plt.hist(samples_std10, bins=100, density=True, histtype='step')

# Make a legend, set limits and show plot
plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.title('Normal Probability Density Function')
plt.xlabel('x')
plt.ylabel('PDF')
plt.show()

# 26. The Normal CDF (Cumulative Distribution Function)

# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)

# Plot CDFs
plt.plot(x_std1, y_std1, marker='.', linestyle='none')
plt.plot(x_std3, y_std3, marker='.', linestyle='none')
plt.plot(x_std10, y_std10, marker='.', linestyle='none')

# Make a legend and show the plot
plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.title('Normal Cumulative Distribution Function')
plt.xlabel('x')
plt.ylabel('CDF')
plt.show()

# 27. The Normal distribution: Properties and warnings

# Are the Belmont Stakes results Normally distributed? Since 1926, the Belmont Stakes is a 1.5 mile-long race of
# 3-year old thoroughbred horses. Secretariat ran the fastest Belmont Stakes in history in 1973. While that was the
# fastest year, 1970 was the slowest because of unusually wet and sloppy conditions. With these two outliers removed
# from the data set, compute the mean and standard deviation of the Belmont winners' times. Sample out of a Normal
# distribution with this mean and standard deviation using the rng.normal() function and plot a CDF. Overlay the ECDF
# from the winning Belmont times. Are these close to Normally distributed?

# Compute mean and standard deviation: mu, sigma
belmont_no_outliers = np.array([148.51, 146.65, 148.52, 150.7, 150.42, 150.88, 151.57, 147.54, 149.65, 148.74
                                   , 147.86, 148.75, 147.5, 148.26, 149.71, 146.56, 151.19, 147.88, 149.16, 148.82
                                   , 148.96, 152.02, 146.82, 149.97, 146.13, 148.1, 147.2, 146., 146.4, 148.2
                                   , 149.8, 147., 147.2, 147.8, 148.2, 149., 149.8, 148.6, 146.8, 149.6
                                   , 149., 148.2, 149.2, 148., 150.4, 148.8, 147.2, 148.8, 149.6, 148.4
                                   , 148.4, 150.2, 148.8, 149.2, 149.2, 148.4, 150.2, 146.6, 149.8, 149.
                                   , 150.8, 148.6, 150.2, 149., 148.6, 150.2, 148.2, 149.4, 150.8, 150.2
                                   , 152.2, 148.2, 149.2, 151., 149.6, 149.6, 149.4, 148.6, 150., 150.6
                                   , 149.2, 152.6, 152.8, 149.6, 151.6, 152.8, 153.2, 152.4, 152.2])

mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)

# Sample out of a normal distribution with this mu and sigma: samples
samples = rng.normal(mu, sigma, size=10000)

# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(belmont_no_outliers)

# Plot the CDFs and show the plot
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.title('Belmont Stakes CDF')
plt.xlabel('Belmont winning time (sec.)')
plt.ylabel('CDF')
plt.show()

# 28. What are the chances of a horse matching or beating Secretariat's record?

# Take a million samples out of the Normal distribution: samples
samples = rng.normal(mu, sigma, size=1000000)

# Compute the fraction that are faster than 144 seconds: prob
prob = np.sum(samples <= 144) / 1000000

# Print the result
print("Probability of besting Secretariat:", prob)

# 29. The Exponential distribution

# Generate 100,000 samples out of an Exponential distribution with parameter beta: samples
samples = rng.exponential(0.1, size=100000)


# If you have a story, you can simulate it!

def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""

    # Draw samples out of first exponential distribution: t1
    t1 = rng.exponential(tau1, size=size)

    # Draw samples out of second exponential distribution: t2
    t2 = rng.exponential(tau2, size=size)

    return t1 + t2


# 30. Distribution of no-hitters and cycles

# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764, 715, size=100000)

# Make the histogram
plt.hist(waiting_times, bins=100, density=True, histtype='step')

# Label axes
plt.title('Distribution of No-Hitters and Cycles')
plt.xlabel('waiting time')
plt.ylabel('PDF')

# Show the plot
plt.show()
