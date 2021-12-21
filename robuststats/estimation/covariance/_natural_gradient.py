'''
File: _natural_gradient.py
File Created: Tuesday, 21st December 2021 11:05:51 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 21st December 2021 11:54:45 am
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''
from pymanopt.function import Callable
from pymanopt import Problem
from pymanopt.manifolds.psd import SymmetricPositiveDefinite
from pymanopt.solvers import SteepestDescent

import numpy as np
import numpy.linalg as la
import autograd
import autograd.numpy as np_a
import autograd.numpy.linalg as a_la

import logging

from .base import get_normalisation_function
from ...models.cost import Tyler_cost_real


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

    @Callable
    def cost(Q):
        return Tyler_cost_real(X, Q, autograd=True)
    return cost


def _generate_Tyler_egrad(X, cost):
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
    dict
        Optimization log from pymanopt
    """
    n_samples, n_features = X.shape
    S = get_normalisation_function(normalisation)
    
    # Normalisation to go to CAE model
    norm_X = la.norm(X, axis=1)
    Y = X / np.tile(norm_X.reshape(n_samples, 1), [1, n_features])

    # Pymanopt setting
    logging.info("Seeting up pymanopt for natural gradient descent")
    manifold = SymmetricPositiveDefinite(n_features)
    cost = _generate_realTyler_cost_function(Y)
    egrad = _generate_Tyler_egrad(Y, cost)
    verbose = verbosity*(verbosity+1)
    problem = Problem(manifold=manifold, cost=cost,
                      egrad=egrad, verbosity=verbose)
    solver = SteepestDescent()
    sigma = solver.solve(problem, x=init)
    return sigma/S(sigma), -cost(sigma), solver._optlog
