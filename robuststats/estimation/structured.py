'''
File: structured.py
File Created: Sunday, 20th June 2021 4:47:02 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 8th July 2021 2:08:42 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import numpy as np
import numpy.linalg as la
import logging

from pymanopt.function import Callable
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, ConjugateGradient
from sklearn.utils.validation import _deprecate_positional_args

from .base import ComplexEmpiricalCovariance
from ..models.probability import complex_multivariate_t
from ..models.manifolds import KroneckerHermitianPositiveElliptical


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
        Y = X - np.tile(loc.reshape((1, n_features)), [n_samples,1])
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


def _estimate_covariance_kronecker_t_gradient(X, df, a, b, mean,
                                              manifold, verbosity=0):
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
            X, self.df, self.a, self.b, self.location_, self.manifold,
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
            X_test, self.df, self.a, self.b, self.location_, self.manifold
        )
        # compute log likelihood
        cost_function = _generate_cost_function(
            X_test, self.df, loc=self.location_
        )
        res = cost_function(A, B)

        return res
