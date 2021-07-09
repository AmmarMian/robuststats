'''
File: structured.py
File Created: Sunday, 20th June 2021 4:47:02 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Friday, 9th July 2021 5:24:54 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import numpy as np
import scipy as sp
import numpy.linalg as la
import logging
from ..utils.verbose import logging_tqdm
from tqdm import tqdm

from pymanopt.function import Callable
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, ConjugateGradient
from sklearn.utils.validation import _deprecate_positional_args

from .base import ComplexEmpiricalCovariance
from ..models.probability import complex_multivariate_t
from ..models.manifolds import KroneckerHermitianPositiveElliptical


def estimation_cov_kronecker_MM(X, a, b, tol=0.001, iter_max=30,
                                       verbosity=False):
    """A function that computes the MM algorithm for Kronecker structured
    covariance matrices with Tyler cost function as presented in:
    >Y. Sun, n_features. Babu and D. n_features. Palomar,
    >"Robust Estimation of Structured Covariance Matrix for Heavy-Tailed
    >Elliptical Distributions,"
    >in IEEE Transactions on Signal Processing,
    >vol. 64, no. 14, pp. 3576-3590, 15 July15, 2016,
    >doi: 10.1109/TSP.2016.2546222.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.
    a : int
        Size of matrix A so that covariance is equal to A \kron B
    b : int
        Size of matrix B so that covariance is equal to A \kron B
    tol : float, optional
        stopping criterion, by default 0.001
    iter_max : int, optional
        number max of iterations, by default 30
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate of A.
    array-like of shape (n_features, n_features)
        estimate of B.
    float
        final error between two iterations.
    int
        number of iterations done.
    """

    n_samples, n_features = X.shape
    Y = X.T

    if a*b != n_features:
        logging.error(
         f"Matrice size incompatible ({a}*{b} != {n_features})"
        )
        return None
    M_i = Y.reshape((b, a, n_samples), order='F')
    delta = np.inf  # Distance between two iterations
    # Initialise estimate to identity
    A = np.eye(a, dtype=Y.dtype)
    i_A = np.linalg.inv(A)
    B = np.eye(b, dtype=Y.dtype)
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)
    else:
        pbar = logging_tqdm(total=iter_max)

    # Recursive algorithm
    while (delta > tol) and (iteration < iter_max):
        # Useful values
        i_B = la.inv(B)
        sqrtm_A = sp.linalg.sqrtm(A)
        sqrtm_B = sp.linalg.sqrtm(B)
        isqrtm_A = la.inv(sqrtm_A)
        isqrtm_B = la.inv(sqrtm_B)

        # Update A with eq. (66)
        M = np.zeros((a, a), dtype=Y.dtype)
        M_numerator = np.zeros((a, a, n_samples), dtype=Y.dtype)
        for i in range(n_samples):
            M_numerator[:, :, i] = M_i[:, :, i].T.conj() @ i_B @  M_i[:, :, i]
            M_denominator = np.trace(i_A@M_numerator[:, :, i])
            M += a/n_samples * (M_numerator[:, :, i]/M_denominator)
        A_new = sqrtm_A @ sp.linalg.sqrtm(isqrtm_A @ M @ isqrtm_A) @ sqrtm_A
        delta_A = la.norm(A_new - A, 'fro') / la.norm(A, 'fro')
        A = A_new
        i_A = la.inv(A)

        # Update B with eq. (67)
        M = np.zeros((b, b), dtype=Y.dtype)
        for i in range(n_samples):
            M_numerator_B = M_i[:, :, i] @ i_A @  M_i[:, :, i].T.conj()
            M_denominator = np.trace(i_A@M_numerator[:, :, i])
            M += b/n_samples * (M_numerator_B/M_denominator)
        B_new = sqrtm_B @ sp.linalg.sqrtm(isqrtm_B @ M @ isqrtm_B) @ sqrtm_B
        delta_B = la.norm(B_new - B, 'fro') / la.norm(B, 'fro')
        B = B_new

        # Condition for stopping
        delta = max(delta_A, delta_B)
        iteration += 1

        if verbosity:
            pbar_v.update()
            pbar_v.set_description(f'(err={delta})', refresh=True)
        else:
            pbar.update()
            pbar.set_description(f'(err={delta})', refresh=True)

    if iteration == iter_max:
        logging.info('Recursive algorithm did not converge')

    return A, B, tol, iteration


class KroneckerEllipticalMM(ComplexEmpiricalCovariance):
    @_deprecate_positional_args
    def __init__(self, a, b, tol=1e-4, iter_max=100,
                 *, store_precision=True, verbosity=False):
        super().__init__(store_precision=store_precision)
        self.a = a
        self.b = b
        self.tol = tol
        self.iter_max = iter_max
        self.verbosity = verbosity

    def fit(self, X, y=None):
        """Fits the kronecker structured covariance according to the
        MM algorithm with Tyler cost function as presented in:
        >Y. Sun, n_features. Babu and D. n_features. Palomar,
        >"Robust Estimation of Structured Covariance Matrix for Heavy-Tailed
        >Elliptical Distributions,"
        >in IEEE Transactions on Signal Processing,
        >vol. 64, no. 14, pp. 3576-3590, 15 July15, 2016,
        >doi: 10.1109/TSP.2016.2546222.

        Assume always centered. Doesn't work otherwhise

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
        """
        X = self._validate_data(X)

        # Using pymanopt solver to obtain estimate
        A, B, _, _ = estimation_cov_kronecker_MM(
            X, self.a, self.b, tol=self.tol, iter_max=self.iter_max,
            verbosity=self.verbosity
        )
        self.A_ = A
        self.B_ = B
        self._set_covariance(np.kron(A, B))

        return self

    def score(self, X_test, y=None):
        """Computes the log-likelihood of a Multivariate t data set with
        `self.covariance_` as an estimator of its covariance matrix.
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.
        """
        logging.error("Sorry, not implemented yet!")
        A, B, _, _ = estimation_cov_kronecker_MM(
            X_test, self.a, self.b, tol=self.tol, iter_max=self.iter_max,
            verbosity=self.verbosity
        )
        # TODO: implement cost function
        res = 0

        return res


def _generate_cost_function(X, df, loc=None):
    """Generate cost function for gradient descent from model parameters and data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.
    df : float
        Degrees of freedom of the distribution.
    loc : array-like of shape (n_features,), optional
        mean of the distribution, by default assumed to be zero.

    Returns
    -------
    callable
        function to compute the cost at given data by X
    """

    @Callable
    def cost(A, B):
        model = complex_multivariate_t(
            loc=loc, shape=np.kron(A, B), df=df
        )
        logpdf_samples = model.logpdf(X)
        return -np.sum(logpdf_samples)

    return cost


def _generate_egrad_function(X, a, b, df, loc=None):
    """Generate gradient direction function for gradient descent
    from model parameters and data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.
    a : int
        Size of matrix A so that covariance is equal to A \kron B
    b : int
        Size of matrix B so that covariance is equal to A \kron B
    loc : array-like of shape (n_features,), optional
        mean of the distribution, by default assumed to be zero.
    df : float
        Degrees of freedom of the distribution.

    Returns
    -------
    callable
        function to compute the euclidian gradient direction at given data by X
    """

    n_samples, n_features = X.shape
    if loc is not None:
        Y = X - np.tile(loc.reshape((1, n_features)), [n_samples, 1])
    else:
        Y = X

    @Callable
    def egrad(A, B):

        # Pre-calculations
        i_A = la.inv(A)
        i_B = la.inv(B)

        grad_A = n_samples*b*i_A
        grad_B = n_samples*a*i_B
        for i in range(n_samples):
            M_i = Y[i, :].reshape((b, a), order='F')
            Q_i = np.real(np.trace(i_A@M_i.conj().T@i_B@M_i))

            grad_A -= (df+a*b)/(df+Q_i) * i_A@M_i.conj().T@i_B@M_i@i_A
            grad_B -= (df+a*b)/(df+Q_i) * i_B@M_i@i_A@M_i.conj().T@i_B

        return (grad_A, grad_B)

    return egrad


def _estimate_covariance_kronecker_t_gradient(X, df, a, b, manifold,
                                              mean=None, verbosity=0):
    """Perform estimation of the covariance of a circular Multivariate t model
    with a structured model of the form: covariance = A \kron B.
    The estimation is done through a conjugate gradient descent method thanks
    to pymanopt.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.
    a : int
        Size of matrix A so that covariance is equal to A \kron B
    b : int
        Size of matrix B so that covariance is equal to A \kron B
    df : float
        Degrees of freedom of the distribution.
    mean : array-like of shape (n_features,), optional
        mean of the distribution, by default assumed to be zero.
    manifold : a KroneckerHermitianPositiveDefiniteStudent object
        pymanopt Manifold object suited to the Kronecker structure parameter
        space.
    verbosity : int
        level of verbosity of pymanopt Problem

    Returns
    -------
    array-like of shape (a, a)
        estimate of A so that covariance = A \kron B
    array-like of shape (b, b)
        estimate of B so that covariance = A \kron B
    """

    cost = _generate_cost_function(X, df, mean)
    egrad = _generate_egrad_function(X, a, b, df, mean)
    problem = Problem(
        manifold=manifold, cost=cost, egrad=egrad, verbosity=verbosity
    )
    solver = ConjugateGradient()
    theta_0 = (np.eye(a, dtype=complex), np.eye(b, dtype=complex))
    A, B = solver.solve(problem, x=theta_0)

    return A, B


class KroneckerStudent(ComplexEmpiricalCovariance):
    @_deprecate_positional_args
    def __init__(self, a, b, df, *, store_precision=True,
                 assume_centered=False, verbosity=0):
        super().__init__(store_precision=store_precision,
                         assume_centered=assume_centered)
        self.a = a
        self.b = b
        self.df = df
        self.alpha = (df+a*b)/(df+a*b+1)
        self.verbosity = verbosity
        self.manifold = KroneckerHermitianPositiveElliptical(a, b, self.alpha)

    def fit(self, X, y=None):
        """Fits the Maximum Likelihood Estimator covariance model
        according to the given training data and parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
        """
        X = self._validate_data(X)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)

        # Using pymanopt solver to obtain estimate
        A, B = _estimate_covariance_kronecker_t_gradient(
            X, self.df, self.a, self.b, self.manifold, self.location_,
            self.verbosity
        )
        self.A_ = A
        self.B_ = B
        self._set_covariance(np.kron(A, B))

        return self

    def score(self, X_test, y=None):
        """Computes the log-likelihood of a Multivariate t data set with
        `self.covariance_` as an estimator of its covariance matrix.
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.
        """
        # compute empirical covariance of the test set
        A, B = _estimate_covariance_kronecker_t_gradient(
            X_test, self.df, self.a, self.b, self.manifold, self.location_,
            self.verbosity
        )
        # compute log likelihood
        cost_function = _generate_cost_function(
            X_test, self.df, loc=self.location_
        )
        res = cost_function(A, B)

        return res
