import numpy as np
from scipy.stats import multivariate_normal


def gaussian_likelihood(
    x: np.ndarray, means: np.ndarray, sigmas: np.ndarray,
) -> np.ndarray:
    """
    Computes the likelihood of each of the datapoints for each of the classes, assuming a
    Gaussian distribution for each of them, with mean and convariance matrix/covariance
    given by 'means' and 'sigmas'.
    Args:
        x (np.ndarray): Datapoints 2D array, rows=samples, columns=features
        means (np.ndarray): Means 2D array, rows=components, columns=features
        sigmas (np.ndarray): Covariance/variance 3D array, dim 0: components,
            dim 1 and 2: n_features x n_features
    Returns:
        (np.ndarray): 2D Gaussian probabilities array, rows=sample, columns=components
    """
    n_components, _ = means.shape
    likelihood = [multivariate_normal.pdf(
        x, means[i, :], sigmas[i, :, :], allow_singular=True) for i in range(n_components)]
    return np.asarray(likelihood).T
