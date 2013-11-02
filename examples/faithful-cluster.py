from pprint import pprint

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from gmm import GMM


# Read in dataset from file
with open('faithful.txt', 'rt') as f:
    data = []
    for row in f:
        cols = row.strip('\r\n').split(' ')
        data.append(np.fromiter(map(lambda x: float(x), cols), np.float))
    data = np.array(data)

# Initialize GMM algorithm
means = [np.array([4.0, 80], np.float), np.array([2.0, 55], np.float)]
covariances = [np.identity(2), np.identity(2)]
mixing_probs = np.array([1/2, 1/2], np.float)
gmm_model = GMM(means, covariances, mixing_probs)

# Fit GMM to the data
gmm_model.fit(data)

# Print fit results
pprint(gmm_model.means)
pprint(gmm_model.covariances)
pprint(gmm_model.mixing_probs)
