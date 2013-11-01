from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

from kmeans.algorithms import KMeans


# Read in dataset from file
with open('faithful.txt', 'rt') as f:
    data = []
    for row in f:
        cols = row.strip('\r\n').split(' ')
        data.append(np.fromiter(map(lambda x: float(x), cols), dtype=np.float))
    data = np.array(data)

# Cluster using K-Means algorithm
n_clusters = 2
k_means = KMeans(n_clusters, tol=1e-8)
k_means.cluster(data[:50])
kmeans_centroids = k_means.centroids

# Label centroids
def label(centroids):
    # Label centroids in an ascending order
    sorted_centroids = np.sort(centroids, axis=0)
    indices = [np.where(centroids == c)[0][0] for c in sorted_centroids]
    return {i: l for i,l in zip(indices, np.arange(n_clusters))}
k_means.label_centroids(label)

# Cluster the data using labeled centroids
data_labeled = k_means.predict(data)

# Plot data with centroids
plt.figure()
c1, c2 = [], []
for d,l in zip(data, data_labeled):
    if l == 0:
        c1.append(d)
    else:
        c2.append(d)
plt.scatter(np.transpose(c1)[0], np.transpose(c1)[1], color='b')
plt.scatter(np.transpose(c2)[0], np.transpose(c2)[1], color='g')
plt.scatter(np.transpose(kmeans_centroids)[0], np.transpose(kmeans_centroids)[1], color='r', marker='x')
plt.savefig('scatter_plot.pdf')
