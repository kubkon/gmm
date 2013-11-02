import numpy as np
import matplotlib.pyplot as plt

from gmm.algorithm import GMM


# Read in dataset from file
with open('faithful.txt', 'rt') as f:
    data = []
    for row in f:
        cols = row.strip('\r\n').split(' ')
        data.append(np.fromiter(map(lambda x: float(x), cols), np.float))
    data = np.array(data)

# Initialize GMM algorithm
means = np.array([np.array([4.0, 80], np.float), np.array([2.0, 55], np.float)])
covariances = np.array([np.identity(3), np.identity(2)])
mixing_probs = np.array([1/2, 1/2], np.float)
gmm_model = GMM(means, covariances, mixing_probs)

# Fit GMM to the data
gmm_model.fit(data)

# Cluster data
labelled = gmm_model.cluster(data)

# Plot clustered data with the location of Gaussian mixtures
plt.figure()

# Plot contours of Gaussian mixtures
for mean, cov in zip(gmm_model.means, gmm_model.covariances):
    # Create grid
    mean_x = mean[0]
    std_x = np.sqrt(cov[0][0])
    mean_y = mean[1]
    std_y = np.sqrt(cov[1][1])
    x = np.linspace(mean_x - 3*std_x, mean_x + 3*std_x, 100)
    y = np.linspace(mean_y - 3*std_y, mean_y + 3*std_y, 100)
    X, Y = np.meshgrid(x, y)

    # Tabulate pdf values
    Z = np.empty(X.shape, np.float)

    for i in np.arange(X.shape[0]):
        for j in np.arange(X.shape[1]):
            v = np.array([X[i][j], Y[i][j]])
            Z[i][j] = gmm_model.multivariate_normal_pdf(v, mean, cov)

    # Plot contours
    plt.contour(X, Y, Z)

# Plot features assigned to each Gaussian mixture
markers = ['o', '+']
colors = ['r', 'b']
for d, l in zip(data, labelled):
    plt.scatter(d[0], d[1], color=colors[l], marker=markers[l])
plt.savefig('scatter_plot.pdf')
