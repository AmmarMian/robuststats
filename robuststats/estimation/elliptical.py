'''
File: elliptical.py
File Created: Sunday, 20th June 2021 8:38:42 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Sunday, 20th June 2021 8:38:46 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import numpy as np
import numpy.linalg as la


from sklearn.utils.validation import _deprecate_positional_args
from pyCovariance.matrix_operators import invsqrtm
from .base import empirical_covariance, ComplexEmpiricalCovariance
from ..models.mappings.complexreal import arraytoreal, arraytocomplex
import logging
from ..utils.verbose import logging_tqdm


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
        Choice between 'trace', 'determinant', 'element', by default None.

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
    elif normalisation is None:
        return (lambda x: 1)
    else:
        logging.error(f'Normalisation type {normalisation} unknown. '
                      'Default to None.')
        return (lambda x: 1)


@_deprecate_positional_args
def tyler_shape_matrix_fixedpoint(X, init=None, tol=1e-4,
                                  iter_max=30, normalisation=None,
                                  **kwargs):
    """Perform Tyler's fixed-point estimation of shape matrix.
    Data is always assumed to be centered
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
        Choice between 'trace', 'determinant', 'element', by default None.
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

        pbar.update()
        pbar.set_description(f'(err={delta})', refresh=True)

    if iteration == iter_max:
        logging.warning('Estimation algorithm did not converge')

    return sigma/S(sigma), delta, iteration


class TylerShapeMatrix(ComplexEmpiricalCovariance):

    def __init__(self, *, method='fixed-point', **kwargs):
        super().__init__()
        self.set_method(method, **kwargs)
        self._err = np.inf
        self._iteration = 0
        self._kwargs = kwargs

    def __str__(self) -> str:
        if len(self._kwargs) > 0:
            kwargs_str = ', '.join(
                [f'{k}={self._kwargs[k]}' for k in self._kwargs])
            return f'TylerShapeMatrix(method={self.method}, ' +\
                kwargs_str + ')'
        else:
            return f'TylerShapeMatrix(method={self.method})'

    def set_method(self, method, **kwargs):
        self.method = method

        # TODO: Add geodesic gradient and BCD
        if method == 'fixed-point':
            def estimation_function(X, init):
                return tyler_shape_matrix_fixedpoint(X, init=init, **kwargs)
        else:
            logging.error("Estimation method not known.")
            raise NotImplementedError(f"Estimation method {method}"
                                      " is not known.")
            estimation_function = None
        self._estimation_function = estimation_function

    def fit(self, X, y=None, init=None):
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
        if self._iteration > 0:
            logging.warning("Overwriting previous fit.")
        X = arraytocomplex(self._validate_data(arraytoreal(X)))
        covariance, err, iteration = self._estimation_function(X, init)
        self._set_covariance(covariance)
        self._err = err
        self._iteration = iteration
        return self

    def score(self, X_test, y=None):
        # TODO: log-likelihood and all for elliptical distributions
        return super().score(X_test)
