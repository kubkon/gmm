import numpy as np


class GMM:
    """
    Implements the expectation-maximisation (EM) algorithm for the
    Gaussian mixture model (GMM). The algorithm is based on the
    pseudo-code described in the book by C. Bishop "Pattern Recognition
    and Machine Learning", chapter 9.
    """
    def __init__(self, means, covariances, mixing_probs):
        self.means = means
        self.covariances = covariances
        self.mixing_probs = mixing_probs
        self.no_components = len(covariances)

    def fit(self, features):
        """
        Fits a GMM into a set of feature data.
        """
        # Initialise
        data_length = features.shape[0]
        norm_densities = np.empty((data_length, self.no_components), np.float)
        responsibilities = np.empty((data_length, self.no_components), np.float)

        # Compute normal densities
        for i in np.arange(data_length):
            x = features[i]

            for j in np.arange(self.no_components):
                norm_densities[i][j] = self.multivariate_normal_pdf(x, self.means[j], self.covariances[j])

        # Estimate log likelihood
        log_likelihood = 0

        for i in np.arange(data_length):
            x = features[i]

            log_likelihood += np.log(np.dot(norm_densities[i], self.mixing_probs))

        # E-step: Evaluate the responsibilities
        for i in np.arange(data_length):
            for j in np.arange(self.no_components):
                numerator = self.mixing_probs[j] * norm_densities[i][j]
                denominator = np.dot(norm_densities[i], self.mixing_probs)
                responsibilities[i][j] = numerator / denominator

        # M-step: Re-estimate the parameters using the current responsibilities
        for i in np.arange(self.no_components):
            pass


    def multivariate_normal_pdf(self, x, mean, covariance):
        """
        Returns normal density value for an n-dimensional random
        vector x.
        """
        centered = x - mean
        cov_inverse = np.linalg.inv(covariance)
        cov_det = np.linalg.det(covariance)
        exponent = np.dot(np.dot(centered.T, cov_inverse), centered)
        return np.exp(-0.5 * exponent) / (np.sqrt(cov_det * np.power(2 * np.pi, self.no_components)))
