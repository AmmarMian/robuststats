'''
File: cost.py
File Created: Tuesday, 30th November 2021 11:19:17 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Wednesday, 8th December 2021 10:17:11 am
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
General use cost functions
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

from ..utils.backend import get_be_autograd, get_be_la_autograd
from ..utils.linalg import invsqrtm
import logging

def Tyler_cost_real(X, Q, autograd=False):
    """Generate cost function for gradient descent for Tyler cost function
    as given in eq. (25) of:
    >Wiesel, A. (2012). Geodesic convexity and covariance estimation.
    >IEEE transactions on signal processing, 60(12), 6182-6189.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.
    Q : array-like of shape (n_features, n_features)
        Shape Matrix.
    autograd : bool
        If true, backend is autograd.numpy. Otherwise it
        is numpy.

    Returns
    -------
    callable
        function to compute the cost at given data by X
    """
    
    if X.shape[1] != Q.shape[0]:
        logging.error(
            f"Incompatible shapes.Shape of X : {X.shape}. Shape of Q: {Q.shape}")
        raise AttributeError(
            f"Incompatible shapes.Shape of X : {X.shape}. Shape of Q: {Q.shape}")

    be = get_be_autograd(autograd)
    be_la = get_be_la_autograd(autograd)

    n, p = X.shape
    temp = invsqrtm(Q, autograd)@X.T
    
    q = be.einsum('ij,ji->i', temp.T, temp)
    return p*be.sum(be.log(q)) + n*be.log(be_la.det(Q))


def Tyler_cost_complex(X, Q, autograd=False):
    """Generate cost function for gradient descent for Tyler cost function
    as given in eq. (25) of:
    >Wiesel, A. (2012). Geodesic convexity and covariance estimation.
    >IEEE transactions on signal processing, 60(12), 6182-6189.
    but for complex valued data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.
    Q : array-like of shape (n_features, n_features)
        Shape Matrix.
    autograd : bool
        If true, backend is autograd.numpy. Otherwise it
        is numpy.

    Returns
    -------
    callable
        function to compute the cost at given data by X
    """
    
    if X.shape[1] != Q.shape[0]:
        logging.error(
            f"Incompatible shapes.Shape of X : {X.shape}. Shape of Q: {Q.shape}")
        raise AttributeError(
            f"Incompatible shapes.Shape of X : {X.shape}. Shape of Q: {Q.shape}")

    be = get_be_autograd(autograd)
    be_la = get_be_la_autograd(autograd)

    n, p = X.shape
    temp = invsqrtm(Q, autograd)@X.T
    q = be.einsum('ij,ji->i', be.conjugate(temp.T), temp)
    return p*be.sum(be.log(be.real(q))) + n*be.log(be.real(be_la.det(Q)))