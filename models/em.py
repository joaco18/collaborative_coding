import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import Union
from scipy.stats import multivariate_normal

logging.basicConfig(level=logging.INFO, format='%(message)s')


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


class ExpectationMaximization():
    def __init__(
        self,
        n_components: int = 3,
        mean_init: Union[str, np.ndarray] = 'random',
        priors: Union[str, np.ndarray] = 'non_informative',
        max_iter: int = 500,
        change_tol: float = 1e-6,
        seed: float = 420,
        verbose: bool = False,
        plot_rate: int = None,
        tissue_models: np.ndarray = None,
        atlas_use: str = None,
        atlas_map: np.ndarray = None,
        previous_result: np.ndarray = None
    ):
        """
        Instatiator of the Expectation Maximization model.
        Args:
            n_components (int, optional): Number of components to be used.
                Defaults to 3.
            mean_init (Union[str, np.ndarray], optional): How to initialize the means.
                You can either pass an array or use one of ['random', 'kmeans'].
                Defaults to 'random'. TODO: Update
            max_iter (int, optional): Maximum number of iterations for the algorith to
                run. Defaults to 100.
            change_tol (float, optional): Minimum change in the summed log-likelihood
                between two iterations, if less stop. Defaults to 1e-5.
            seed (float, optional): Seed to guarantee reproducibility. Defaults to 420.
            verbose (bool, optional): Whether to print messages on evolution or not.
                Defaults to False.
            plot_rate (int, optional): Number of iterations after which a scatter plot
                (or a histogram in 1D data) is plotted to see the progress in classification.
                Defaults to None, which means no plotting.
            tissue_models (np.ndarray, optional): Tissue probability maps for the different
                components. This will be used for generating the initialization of the EM
                algorithm. Dimesions should be [n_components, n_hist_bins]
                If not used, None should be passed. Defaults to None
            atlas_use (str, optional): How to use the probability maps, either at the end
                ('after') or in each iteration ('into'). Defaults to None which means atlas
                are not used in the EM iterative process.
            atlas_map (np.ndarray, optional): Atlas volumes, it should have dimesions
                [n_components, [volume dimensions]]. The atlas map for each component can be
                either binary or a probability one. Defaults to None, which means not using
                the atlas_maps
            previous_result (np.ndarray, optional): Result from EM part for "after" use of atlases.
                If provided just the last multiplication is done. Defaults to None.
        """
        self.n_components = n_components
        self.mean_init = mean_init
        self.priors = priors
        self.change_tol = change_tol
        self.previous_result = previous_result
        self.max_iter = max_iter
        self.seed = seed
        self.verbose = verbose
        self.plot_rate = plot_rate
        self.fitted = False
        self.cluster_centers_ = None
        self.n_iter_ = 0
        self.tissue_models = tissue_models
        self.atlas_use = atlas_use
        self.atlas_map = atlas_map
        if self.atlas_map is not None:
            self.atlas_map = self.atlas_map.T
            self.atlas_map[np.sum(self.atlas_map, axis=1) == 0, :] = np.ones((1, 3))/3
            self.atlas_map = self.atlas_map / self.atlas_map.sum(axis=1)[:, None]
        self.training = False

        # Check kind of means to be used
        mean_options = ['random', 'kmeans', 'tissue_models', 'mni_atlas', 'mv_atlas']
        if isinstance(mean_init, str) and (mean_init not in mean_options):
            raise Exception(
                "Initial means must be either 'random', 'kmeans', 'tisssue_models', "
                "'label_prop' or an array of 'n_components' rows, and n_features number of columns"
            )

    def fit(self, x: np.ndarray):
        """ Runs the EM procedure using the data provided and the configured parameters.
        Args:
            x (np.ndar ray): Datapoints 2D array, rows=samples, columns=features
        """
        self.training = True
        self.x = x
        self.n_feat = x.shape[1] if np.ndim(x) > 1 else 1
        self.n_samples = len(x)
        self.labels = np.ones((self.n_samples, self.n_components))
        self.posteriors = np.zeros((self.n_samples, self.n_components))

        # Define kind of means to be used
        self.mean_type = 'Passed array'
        if isinstance(self.mean_init, str):
            if self.mean_init == 'random':
                self.mean_type = 'Random Init'
                rng = np.random.default_rng(seed=self.seed)
                idx = rng.choice(self.n_samples, size=self.n_components, replace=False)
                self.posteriors[idx, np.arange(self.n_components)] = 1
                self.priors = np.ones((self.n_components, 1)) / self.n_components
            elif self.mean_init == 'kmeans':
                self.mean_type = 'K-Means'
                kmeans = KMeans(
                    n_clusters=self.n_components, random_state=self.seed).fit(self.x)
                self.posteriors[np.arange(self.n_samples), kmeans.labels_] = 1
            elif self.mean_init == 'tissue_models':
                self.mean_type = 'Tissue Models'
                tissue_prob_maps = np.zeros((self.n_samples, self.n_components))
                for c in range(self.n_components):
                    tissue_prob_maps[:, c] = self.tissue_models[c, :][self.x[:, 0]]
                self.posteriors = tissue_prob_maps
                self.posteriors[np.arange(self.n_samples), np.argmax(tissue_prob_maps, axis=1)] = 1
            elif self.mean_init == 'mni_atlas':
                self.mean_type = 'Label Propagation MNI Atlas'
                self.posteriors = self.atlas_map
            else:
                self.mean_type = 'Label Propagation Medvision Atlas'
                self.posteriors = self.atlas_map
        else:
            self.means = self.mean_init
            idx = rng.choice(self.n_samples, size=self.n_components, replace=False)
            self.labels[idx, np.arange(self.n_components)] = 1

        if self.previous_result is not None:
            self.posteriors = self.previous_result
            if self.mean_init == 'kmeans':
                self.match_labels()
            self.posteriors = self.posteriors * self.atlas_map
            self.maximization(initial=True, initial_random=(self.mean_init == 'random'))
        else:
            # Do initial maximization step to get gaussian inital parameters
            self.maximization(initial=True, initial_random=(self.mean_init == 'random'))

            # Log initial info
            if self.verbose:
                logging.info('Starting Expectation Maximization Algorithm')
                logging.info(f'Priors type: {self.priors_type} \n {self.priors}')
                logging.info(f'Mean type: {self.mean_type} \n {self.means}')

            # Expectation Maximization process
            self.expectation_maximization()
        self.cluster_centers_ = self.means
        self.fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Predicts the datopoints in x according to the gaussians found runing the
        EM fitting process.
        Args:
            x (np.ndarray): Datapoints 2D array, rows=samples, columns=features
        Returns:
            (np.ndarray): One-hot predictions 2D array, rows=samples, columns=components
        """
        self.training = False
        self.x = x
        if not self.fitted:
            raise Exception('Algorithm hasn\'t been fitted')
        self.expectation()
        self.predictions = np.argmax(self.posteriors, 1)
        return self.posteriors, self.predictions

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the datopoints in x according to the gaussians found furing the
        EM fitting process. Returns the posterior probability for each point for
        each class.
        Args:
            x (np.ndarray): Datapoints 2D array, rows=samples, columns=features
        Returns:
            (np.ndarray): Posterior probabilities 2D array, rows=samples,
                columns=components
        """
        self.x = x
        self.training = False
        if not self.fitted:
            raise Exception('Algorithm hasn\'t been fitted')
        self.expectation()
        return self.posteriors

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        """ Runs the EM procedure using the data provided and the configured parameters and
        predicts the datopoints in x according to the reuslting gaussians.
        Args:
            x (np.ndarray): Datapoints 2D array, rows=samples, columns=features
        Returns:
            np.ndarray: One-hot predictions 2D array, rows=samples, columns=components
        """
        self.fit(x)
        self.posteriors, self.predictions = self.predict(self.x)
        return self.posteriors, self.predictions

    def expectation_maximization(self):
        """ Expectation Maximization process """
        # Initialize log likelihood
        prev_log_lkh = 0

        # Run EM iterations
        description = f'EM-{self.mean_init}-{self.atlas_use}'
        for it in tqdm(range(self.max_iter), disable=self.verbose, desc=description):
            self.n_iter_ = it + 1

            # E-step
            self.expectation(it)

            # Scatter plots to see the evolution
            if self.plot_rate is not None:
                self.plots(it)

            # Check stop criteria: tolerance over log-likelihood
            for i in range(self.n_components):
                self.likelihood[:, i] = self.likelihood[:, i] * self.priors[i]
            log_lkh = np.sum(np.log(np.sum(self.likelihood, 1)), 0)
            difference = abs(prev_log_lkh - log_lkh)
            prev_log_lkh = log_lkh
            if difference < self.change_tol:
                if self.atlas_use == 'after':
                    self.posteriors = self.posteriors * self.atlas_map
                break

            # M-Step
            self.maximization()
            if self.verbose:
                logging.info(f'Iteration: {it} - Log likelihood change: {difference}')

    def expectation(self, it: int = None):
        """ Expectation Step:
        Obtains the likelihoods with the current means and covariances, and computes the
        posterior probabilities (or weights)
        """
        self.likelihood = gaussian_likelihood(self.x, self.means, self.sigmas)
        num = np.asarray([
            self.likelihood[:, j] * self.priors[j] for j in range(self.n_components)]).T
        denom = np.sum(num, 1)
        self.posteriors = np.asarray([num[:, j] / denom for j in range(self.n_components)]).T

        if (self.atlas_use == 'into' and self.training):
            if (it == 0) and (self.mean_init == 'kmeans'):
                self.match_labels()
            self.posteriors = self.posteriors * self.atlas_map

        if (self.atlas_use == 'after' and not(self.training)):
            if self.mean_init == 'kmeans':
                self.match_labels()
            self.posteriors = self.posteriors * self.atlas_map

    def match_labels(self):
        labels_em = np.argmax(self.posteriors, axis=1)
        labels_atlas = np.argmax(self.atlas_map, axis=1)
        order = {}
        for label in np.unique(labels_atlas):
            labels, counts = np.unique(labels_em[labels_atlas == label], return_counts=True)
            order[label] = labels[np.argmax(counts)]
        self.posteriors_ = self.posteriors.copy()
        for key, val in order.items():
            self.posteriors_[key, :] = self.posteriors[val, :]

    def maximization(self, initial: bool = False, initial_random: bool = False):
        """ Maximization Step:
        With the belonging of each point to certain class -given by the posterior wieght-
        computes the new mean and covariance for each class
            initial_random (str, optional): In random initialization, start covariance
                matrices the same one for all components. Defaults to None
        """
        # Redefine labels with maximum a posteriori
        self.labels = np.zeros((self.x.shape[0], self.n_components))
        self.labels[np.arange(self.n_samples), np.argmax(self.posteriors, axis=1)] = 1

        # Get means
        self.counts = np.sum(self.posteriors, 0)
        weithed_avg = np.dot(self.posteriors.T, self.x)

        # Get means
        self.means = weithed_avg / self.counts[:, np.newaxis]

        # Get covariances
        self.sigmas = np.zeros((self.n_components, self.n_feat, self.n_feat))
        if initial_random:
            sigma = np.cov((self.x - np.mean(self.x, axis=0)).T)
            for i in range(self.n_components):
                self.sigmas[i] = sigma
                self.sigmas[i].flat[:: self.n_feat + 1] += 1e-6
            if np.ndim(self.sigmas) == 1:
                self.sigmas = (self.sigmas[:, np.newaxis])[:, np.newaxis]
        else:
            for i in range(self.n_components):
                diff = self.x - self.means[i, :]
                weighted_diff = self.posteriors[:, i][:, np.newaxis] * diff
                self.sigmas[i] = np.dot(weighted_diff.T, diff) / self.counts[i]
            self.priors = self.counts / len(self.x)

    def plots(self, it: int):
        """ Plots the scatter plots (or histograms in 1D cases) of data assignments
        to each gaussian across the iterations.
        Args:
            it (int): Iteration number.
        """
        # plt.ioff()
        if (it % self.plot_rate) == 0:
            predictions = np.argmax(self.posteriors, 1)
            if self.n_feat == 1:
                _, ax = plt.subplots()
                sns.histplot(
                    x=self.x[:, 0], hue=predictions, kde=False, bins=255,
                    stat='probability', ax=ax)
                plt.xlabel('Intensities')
                plt.title(f'Labels assignment at iteration {it}')
                plt.show(block=False)
                plt.pause(0.01)
            else:
                plt.figure()
                if it == 0:
                    plt.scatter(x=self.x[:, 0], y=self.x[:, 1])
                for i in range(self.n_components):
                    plt.scatter(
                        x=self.x[self.labels[:, i] == 1, 0],
                        y=self.x[self.labels[:, i] == 1, 1]
                    )
                plt.ylabel('T2 intensities')
                plt.xlabel('T1 intensities')
                plt.title(f'Labels assignment at iteration {it}')
                sns.despine()
                plt.show(block=False)
                plt.pause(0.01)
                if it == 0:
                    plt.figure()
                    indx = np.random.choice(np.arange(self.x.shape[0]), 1000, False)
                    sample = self.x[indx, :]
                    sns.kdeplot(x=sample[:, 0], y=sample[:, 1])
                    plt.show(block=False)
                    plt.pause(0.01)
