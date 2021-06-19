'''
File: structuredcovariance.py
Created Date: Saturday June 19th 2021 - 06:20pm
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Sat Jun 19 2021
Modified By: Ammar Mian
-----
Copyright (c) 2021 Université Savoie Mont-Blanc
'''

import numpy as np
from numpy import linalg as la

from pymanopt.manifolds.hpd import HermitianPositiveDefinite,\
     SpecialHermitianPositiveDefinite
from pymanopt.tools.multi import multitransp
from pyCovariance.features.base import Product


class KroneckerHermitianPositiveElliptical(Product):
    """Manifold of (n x n)^k complex Hermitian positive definite matrices with
    the model: M = A kron B where A and B are complex Hermitian positive
    definite matrices of size (a x a) and (b x b) such that ab = n and
    det(A)=1.


    Attributes
    ----------
    _n : int
        Size of covariance matrix.
    _a : int
        Size of matrix A.
    _b : int
        Size of matrix B.
    _alpha : float
        Coeficient \alpha corresponding to the elliptical model.
    _k : int, optional
        Number of covariance matrices. Default is 1.
    """

    def __init__(self, n, a, b, alpha, k=1):
        if n != a*b:
            raise AttributeError(
                f'Size of matrices incompatible: {a} x {b} != {n}')
        self._n = n
        self._a = a
        self._b = b
        self._k = k
        self._alpha = alpha
        weights = (self._alpha*self._b, self._alpha*self._a)
        manifolds = (SpecialHermitianPositiveDefinite(a, k),
                     HermitianPositiveDefinite(b, k))
        super().__init__(manifolds, weights)

    def inner(self, theta, xi, eta):
        """Inner product on kronecker product manifold of shpd and hpd
        matrices according to Fisher information metric associated with
        an elliptical distribution distribution.

        Parameters
        ----------
        theta : list
            list of two array_like corresponding to the reference point.
        xi : list
            list two array_like for each component of the tangent vector
        eta : list
            list two array_like for each component of the tangent vector

        Returns
        -------
        float
            the inner product.
        """
        _part_A = self._weights[0]*self._manifolds[0].inner(theta[0],
                                                            xi[0], eta[0])
        _part_B_xi = la.solve(theta[1], xi[1])
        _part_B_eta = la.solve(theta[1], eta[1])
        _part_B = self._weights[1]*np.real(
                    np.tensordot(_part_B_xi, multitransp(_part_B_eta),
                                 axes=theta[1].ndim)) +\
            (self._alpha - 1) * (self._a**2) *\
            np.real(np.trace(_part_B_xi)*np.trace(_part_B_eta))
        return _part_A + _part_B
