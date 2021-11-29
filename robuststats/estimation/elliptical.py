'''
File: elliptical.py
File Created: Sunday, 20th June 2021 8:38:42 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Monday, 29th November 2021 3:56:37 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import numpy as np
import numpy.linalg as la
import logging

from sklearn.utils.validation import _deprecate_positional_args
from sklearn.covariance import EmpiricalCovariance, empirical_covariance
from .base import complex_empirical_covariance, ComplexEmpiricalCovariance,\
                RealEmpiricalCovariance
from ..models.mappings import arraytoreal, arraytocomplex

from ..utils.verbose import logging_tqdm
from ..utils.linalg import invsqrtm
from tqdm import tqdm

from pymanopt.function import Callable
from pymanopt import Problem
from pymanopt.manifolds.psd import SymmetricPositiveDefinite
from pymanopt.solvers import SteepestDescent
import autograd
import autograd.numpy as a_np
import autograd.numpy.linalg as a_la

def get_normalisation_function(normalisation=None):
    """Get normalisation function to perform for shape matrix
    as defined in subsection 3.1 of:
    >Marc Hallin, Davy Paindaveine,
    >Optimal tests for homogeneity of covariance, scale, and shape,
    >Journal of Multivariate Analysis,
    >Volume 100, Issue 3, 2009

    Parameters
    ----------
    normalisaton : string, optional
        type of normalisation to perform.
        Choice between 'trace', 'determinant', 'element'
        and 'None', by default 'None'.

    Returns
    -------
    function
        normalisation function.
    """
    def trace(covariance):
        return np.real(np.trace(covariance)) / covariance.shape[0]

    def determinant(covariance):
        return np.real(la.det(covariance))**(1/covariance.shape[0])

    def element(covariance):
        return covariance[0, 0]

    if normalisation == 'trace':
        return trace
    elif normalisation == 'determinant':
        return determinant
    elif normalisation == 'element':
        return element
    elif normalisation == 'None':
        return (lambda x: 1)
    else:
        logging.error(f'Normalisation type {normalisation} unknown. '
                      'Default to None.')
        return (lambda x: 1)


# -----------------------------------------------------------------------------
# Real-valued estimators
# -----------------------------------------------------------------------------
def _generate_realTyler_cost_function(X):
    """Generate cost function for gradient descent for Tyler cost function
    as given in eq. (25) of:
    >Wiesel, A. (2012). Geodesic convexity and covariance estimation.
    >IEEE transactions on signal processing, 60(12), 6182-6189.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.

    Returns
    -------
    callable
        function to compute the cost at given data by X
    """

    n, p = X.shape

    @Callable
    def cost(Q):
        temp = invsqrtm(Q)@X.T
        q = a_np.einsum('ij,ji->i', temp.T, temp)
        return p*a_np.sum(a_np.log(q)) + n*a_np.log(a_la.det(Q))

    return cost


def _generate_realTyler_egrad(X, cost):
    """Generate euclidean gradient corresponding to Tyler cost function.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.
    cost :
        cost_function depending on the data X and obtained from
        _generate_realTyler_cost_function.

    Returns
    -------
    callable
        function to compute the cost at given data by X
    """

    n, p = X.shape

    @Callable
    def egrad(Q):
        res = autograd.grad(cost, argnum=[0])(Q)
        return res[0]

    return egrad


@_deprecate_positional_args
def tyler_shape_matrix_naturalgradient(X, init=None, normalisation='None',
                                  verbosity=False, **kwargs):
    """Perform Tyler's estimation of shape matrix with natural 
    gradient on SPD manifold using pymanopt.
    Data is always assumed to be centered.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    normalisaton : string, optional
        type of normalisation to perform.
        Choice between 'trace', 'determinant', 'element'
        and 'None', by default 'None'.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate.
    float
        final error between two iterations.
    int
        number of iterations done.
    """
    n_samples, n_features = X.shape
    S = get_normalisation_function(normalisation)

    # Pymanopt setting
    logging.info("Seeting up pymanopt for natural gradient descent")
    manifold = SymmetricPositiveDefinite(n_features)
    cost = _generate_realTyler_cost_function(X)
    egrad = _generate_realTyler_egrad(X, cost)
    verbose = verbosity*(verbosity+1)
    problem = Problem(manifold=manifold, cost=cost,
                      egrad=egrad, verbosity=verbose)
    solver = SteepestDescent()
    sigma = solver.solve(problem, x=init)
    return sigma/S(sigma), -cost(sigma), None


@_deprecate_positional_args
def tyler_shape_matrix_fixedpoint(X, init=None, tol=1e-4,
                                  iter_max=30, normalisation='None',
                                  verbosity=False, **kwargs):
    """Perform Tyler's fixed-point estimation of shape matrix.
    Data is always assumed to be centered.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    tol : float, optional
        tolerance for convergence of algorithm, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations, by default 30.
    normalisaton : string, optional
        type of normalisation to perform.
        Choice between 'trace', 'determinant', 'element'
        and 'None', by default 'None'.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate.
    float
        final error between two iterations.
    int
        number of iterations done.
    """

    # Initialisation
    N, p = X.shape
    if init is None:
        sigma = empirical_covariance(X)
    else:
        sigma = init

    S = get_normalisation_function(normalisation)

    delta = np.inf  # Distance between two iterations
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)
    else:
        pbar = logging_tqdm(total=iter_max)
    while (delta > tol) and (iteration < iter_max):
        # compute expression of Tyler estimator
        temp = invsqrtm(sigma)@X.T
        tau = np.einsum('ij,ji->i', temp.T, temp)
        tau = (1/p) * tau
        temp = X.T / np.sqrt(tau)
        sigma_new = (1/N) * temp@temp.T

        # condition for stopping
        delta = la.norm(sigma_new - sigma) / la.norm(sigma)
        iteration += 1

        # updating sigma
        sigma = sigma_new

        if verbosity:
            pbar_v.update()
            pbar_v.set_description(f'(err={delta})', refresh=True)
        else:
            pbar.update()
            pbar.set_description(f'(err={delta})', refresh=True)

    if verbosity:
        pbar_v.close()
    else:
        pbar.close()

    if iteration == iter_max:
        logging.warning('Estimation algorithm did not converge')

    return sigma/S(sigma), delta, iteration


class TylerShapeMatrix(RealEmpiricalCovariance):
    """Tyler M-estimator of shape matrix with real values.
    See:
    >David E. Tyler.
    >"A Distribution-Free M-Estimator of Multivariate Scatter."
    >Ann. Statist. 15 (1) 234 - 251, March, 1987.
    >https://doi.org/10.1214/aos/1176350263

    Parameters
    ----------
    tol : float, optional
        criterion for error between two iterations, by default 1e-4.
    method : str, optional
        way to compute the solution between
        'fixed-point', 'bcd' or 'gradient', by default 'fixed-point'.
    iter_max : int, optional
        number of maximum iterations of algorithm, by default 100.
    normalisation : str, optional
        type of normalisation between 'trace', 'determinant'
        or 'None', by default 'None'.
    verbosity : bool, optional
        show a progressbar of algorithm, by default False.
    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    err_ : float
        final error between two iterations.
    iteration_ : int
        number of iterations done for estimation.
    """

    def __init__(self, tol=1e-4, method='fixed-point',
                 iter_max=100, normalisation='None',
                 verbosity=False):
        super().__init__()
        self.method = method
        self.verbosity = verbosity
        self.tol = tol
        self.iter_max = iter_max
        self.normalisation = normalisation

        self.set_method()
        self.err_ = np.inf
        self.iteration_ = 0

    def set_method(self):
        # TODO: Add geodesic gradient and BCD
        if self.method == 'fixed-point':
            def estimation_function(X, init, **kwargs):
                return tyler_shape_matrix_fixedpoint(
                    X, init=init, tol=self.tol, iter_max=self.iter_max,
                    normalisation=self.normalisation,
                    verbosity=self.verbosity, **kwargs)
        elif self.method == 'natural gradient':
            def estimation_function(X, init, **kwargs):
                return tyler_shape_matrix_naturalgradient(
                    X, init=init, normalisation=self.normalisation,
                    verbosity=self.verbosity, **kwargs)
        else:
            logging.error("Estimation method not known.")
            raise NotImplementedError(f"Estimation method {self.method}"
                                      " is not known.")

        self._estimation_function = estimation_function

    def fit(self, X, y=None, init=None, **kwargs):
        """Fits the Tyler estimator of shape matrix with the
        specified method when initialised object.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        init : array-like of shape (n_features, n_features), optional
            Initial point to start the estimation.
        Returns
        -------
        self : object
        """
        if self.iteration_ > 0 and self.verbosity:
            logging.warning("Overwriting previous fit.")
        X = self._validate_data(X)
        covariance, err, iteration = self._estimation_function(
            X, init, **kwargs)
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None, model='Gaussian'):
        # TODO: log-likelihood and all for elliptical distributions
        if model == 'Gaussian':
            return super().score(X_test)
        else:
            raise NotImplementedError(
                f"Model {model} is not known.")


# -----------------------------------------------------------------------------
# Complex-valued estimators
# -----------------------------------------------------------------------------
@_deprecate_positional_args
def complex_tyler_shape_matrix_fixedpoint(X, init=None, tol=1e-4,
                                  iter_max=30, normalisation='None',
                                  verbosity=False, **kwargs):
    """Perform Tyler's fixed-point estimation of shape matrix.
    Data is always assumed to be centered and complex.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    tol : float, optional
        tolerance for convergence of algorithm, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations, by default 30.
    normalisaton : string, optional
        type of normalisation to perform.
        Choice between 'trace', 'determinant', 'element'
        and 'None', by default 'None'.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate.
    float
        final error between two iterations.
    int
        number of iterations done.
    """

    # Initialisation
    N, p = X.shape
    if init is None:
        sigma = complex_empirical_covariance(X)
    else:
        sigma = init

    S = get_normalisation_function(normalisation)

    delta = np.inf  # Distance between two iterations
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)
    else:
        pbar = logging_tqdm(total=iter_max)
    while (delta > tol) and (iteration < iter_max):
        # compute expression of Tyler estimator
        temp = invsqrtm(sigma)@X.T
        tau = np.einsum('ij,ji->i', temp.conj().T, temp)
        tau = (1/p) * np.real(tau)
        temp = X.T / np.sqrt(tau)
        sigma_new = (1/N) * temp@temp.conj().T

        # condition for stopping
        delta = la.norm(sigma_new - sigma) / la.norm(sigma)
        iteration += 1

        # updating sigma
        sigma = sigma_new

        if verbosity:
            pbar_v.update()
            pbar_v.set_description(f'(err={delta})', refresh=True)
        else:
            pbar.update()
            pbar.set_description(f'(err={delta})', refresh=True)

    if verbosity:
        pbar_v.close()
    else:
        pbar.close()

    if iteration == iter_max:
        logging.warning('Estimation algorithm did not converge')

    return sigma/S(sigma), delta, iteration


class ComplexTylerShapeMatrix(ComplexEmpiricalCovariance):
    """Tyler M-estimator of shape matrix with complex values.
    See:
    >David E. Tyler.
    >"A Distribution-Free M-Estimator of Multivariate Scatter."
    >Ann. Statist. 15 (1) 234 - 251, March, 1987.
    >https://doi.org/10.1214/aos/1176350263

    Parameters
    ----------
    tol : float, optional
        criterion for error between two iterations, by default 1e-4.
    method : str, optional
        way to compute the solution between
        'fixed-point', 'bcd' or 'gradient', by default 'fixed-point'.
    iter_max : int, optional
        number of maximum iterations of algorithm, by default 100.
    normalisation : str, optional
        type of normalisation between 'trace', 'determinant'
        or 'None', by default 'None'.
    verbosity : bool, optional
        show a progressbar of algorithm, by default False.
    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    err_ : float
        final error between two iterations.
    iteration_ : int
        number of iterations done for estimation.
    """

    def __init__(self, tol=1e-4, method='fixed-point',
                 iter_max=100, normalisation='None',
                 verbosity=False):
        super().__init__()
        self.method = method
        self.verbosity = verbosity
        self.tol = tol
        self.iter_max = iter_max
        self.normalisation = normalisation

        self.set_method()
        self.err_ = np.inf
        self.iteration_ = 0

    def set_method(self):
        # TODO: Add geodesic gradient and BCD
        if self.method == 'fixed-point':
            def estimation_function(X, init, **kwargs):
                return complex_tyler_shape_matrix_fixedpoint(
                    X, init=init, tol=self.tol, iter_max=self.iter_max,
                    normalisation=self.normalisation,
                    verbosity=self.verbosity, **kwargs)
        else:
            logging.error("Estimation method not known.")
            raise NotImplementedError(f"Estimation method {self.method}"
                                      " is not known.")

        self._estimation_function = estimation_function

    def fit(self, X, y=None, init=None, **kwargs):
        """Fits the Tyler estimator of shape matrix with the
        specified method when initialised object.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        init : array-like of shape (n_features, n_features), optional
            Initial point to start the estimation.
        Returns
        -------
        self : object
        """
        if self.iteration_ > 0 and self.verbosity:
            logging.warning("Overwriting previous fit.")
        X = self._validate_data(X)
        covariance, err, iteration = self._estimation_function(
            X, init, **kwargs)
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None, model='Gaussian'):
        # TODO: log-likelihood and all for elliptical distributions
        if model == 'Gaussian':
            return super().score(X_test)
        else:
            raise NotImplementedError(
                f"Model {model} is not known.")
