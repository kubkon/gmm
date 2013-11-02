import numpy as np


class GMM:
    """
    Implements the expectation-maximisation (EM) algorithm for the
    Gaussian mixture model (GMM). The algorithm is based on the
    pseudo-code described in the book by C. Bishop "Pattern Recognition
    and Machine Learning", chapter 9.
    """
    def __init__(self, means, covariances, mixing_probs, epsilon=1e-6):
        self.means = means
        self.covariances = covariances
        self.mixing_probs = mixing_probs
        self.no_components = len(covariances)
        self.epsilon = epsilon

    def fit(self, features):
        """
        Fits a GMM into a set of feature data.
        """
        # Initialise
        n, _ = features.shape
        norm_densities = np.empty((n, self.no_components), np.float)
        responsibilities = np.empty((n, self.no_components), np.float)
        old_log_likelihood = 0

        while True:

            # Compute normal densities
            for i in np.arange(n):
                x = features[i]
                for j in np.arange(self.no_components):
                    norm_densities[i][j] = self.multivariate_normal_pdf(x, self.means[j], self.covariances[j])

            # Estimate log likelihood
            log_vector = np.log(np.array([np.dot(self.mixing_probs.T, norm_densities[i]) for i in np.arange(n)]))
            log_likelihood = np.dot(log_vector.T, np.ones(n))
            
            # Check for convergence
            if np.absolute(log_likelihood - old_log_likelihood) < self.epsilon:
                break

            # E-step: evaluate responsibilities
            for i in np.arange(n):
                x = features[i]
                denominator = np.dot(self.mixing_probs.T, norm_densities[i])
                for j in np.arange(self.no_components):
                    responsibilities[i][j] = self.mixing_probs[j] * norm_densities[i][j] / denominator

            # M-step: re-estimate the parameters
            for i in np.arange(self.no_components):
                responsibility = (responsibilities.T)[i]
                # Common denominator
                denominator = np.dot(responsibility.T, np.ones(n))
                # Update mean
                self.means[i] = np.dot(responsibility.T, features) / denominator
                # Update covariance
                difference = features - np.tile(self.means[i], (n, 1))
                self.covariances[i] = np.dot(np.multiply(responsibility.reshape(n,1), difference).T, difference) / denominator
                # Update mixing probabilities
                self.mixing_probs[i] = denominator / n

            old_log_likelihood = log_likelihood


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
