import numpy as np
import scipy.stats as stats


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

    def fit(self, features):
        """
        Fits a GMM into a set of feature data.
        """
        pass
