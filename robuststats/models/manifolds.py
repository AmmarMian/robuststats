'''
File: manifolds.py
Created Date: Saturday June 19th 2021 - 06:20pm
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Monday, 29th November 2021 11:42:50 am
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Many parts are done by Antoine Collas in his fork of pymanopt:
https://github.com/antoinecollas/pymanopt
-----
Copyright (c) 2021 Université Savoie Mont-Blanc
'''

import numpy as np
from numpy import linalg as la, random as rnd
from scipy.linalg import sqrtm

from pymanopt.tools.multi import multitransp
import pymanopt.manifolds as man
from pymanopt.manifolds.product import _ProductTangentVector
from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold
from pymanopt.tools.multi import multihconj, multilog
from pymanopt.tools.multi import multiprod, multitransp
from ..utils.linalg import multiherm

# ---------------------------------------------------------------------
# REAL MANIFOLDS
# ---------------------------------------------------------------------
class StrictlyPositiveVectors(EuclideanEmbeddedSubmanifold):
    """Manifold of k strictly positive n-dimensional vectors, denoted ((R++)^n)^k.
    Since ((R++)^n)^k is isomorphic to
    (D_n^{++})^k (manifold of positive definite diagonal matrices of size n),
    the geometry is inherited of the positive definite matrices.
    """
    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        if k == 1:
            name = ("Manifold of strictly positive vectors of size {}").format(
                n)
        else:
            name = ("Product manifold of {} \
                    strictly positive vectors of size {}").format(k, n)
        dimension = int(k * n)
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def inner(self, x, u, v):
        inv_x = (1./x)
        return np.sum(inv_x*u*inv_x*v, axis=0, keepdims=True)

    def proj(self, x, u):
        return u

    def norm(self, x, u):
        return np.sqrt(self.inner(x, u, u))

    def rand(self):
        return rnd.uniform(low=1e-6, high=1, size=(self._n, self._k))

    def randvec(self, x):
        u = rnd.randn(self._n, self._k)
        return u / self.norm(x, u)

    def zerovec(self, x):
        return np.zeros((self._n, self._k))

    def dist(self, x, y):
        return la.norm(np.log(x)-np.log(y), axis=0, keepdims=True)

    def egrad2rgrad(self, x, u):
        return u*(x**2)

    def ehess2rhess(self, x, egrad, ehess, u):
        return ehess*(x**2) + egrad*u*x

    def exp(self, x, u):
        return x*np.exp((1./x)*u)

    def retr(self, x, u):
        return x + u + (1/2)*(x**-1)*(u**2)

    def log(self, x, y):
        return x*np.log((1./x)*y)

    def transp(self, x1, x2, d):
        res = self.proj(x2, x2*(x1**(-1)*d))
        return res

# ---------------------------------------------------------------------
# COMPLEX MANIFOLDS
# ---------------------------------------------------------------------
class HermitianPositiveDefinite(EuclideanEmbeddedSubmanifold):
    """Manifold of (n x n)^k complex Hermitian positive definite matrices.
    """
    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        if k == 1:
            name = ("Manifold of Hermitian positive definite\
                    ({} x {}) matrices").format(n, n)
        else:
            name = "Product manifold of {} ({} x {}) matrices".format(k, n, n)
        dimension = 2 * int(k * n * (n + 1) / 2)
        super().__init__(name, dimension)

    def rand(self):
        # Generate eigenvalues between 1 and 2
        # (eigenvalues of a symmetric matrix are always real).
        d = np.ones((self._k, self._n, 1)) + rnd.rand(self._k, self._n, 1)

        # Generate an orthogonal matrix. Annoyingly qr decomp isn't
        # vectorized so need to use a for loop. Could be done using
        # svd but this is slower for bigger matrices.
        u = np.zeros((self._k, self._n, self._n), dtype=np.complex)
        for i in range(self._k):
            u[i], r = la.qr(
                rnd.randn(self._n, self._n)+1j*rnd.randn(self._n, self._n))

        if self._k == 1:
            return multiprod(u, d * multihconj(u))[0]
        return multiprod(u, d * multihconj(u))

    def randvec(self, x):
        k = self._k
        n = self._n
        if k == 1:
            u = multiherm(rnd.randn(n, n)+1j*rnd.randn(n, n))
        else:
            u = multiherm(rnd.randn(k, n, n)+1j*rnd.randn(k, n, n))
        return u / self.norm(x, u)

    def zerovec(self, x):
        k = self._k
        n = self._n
        if k != 1:
            return np.zeros((k, n, n), dtype=np.complex)
        return np.zeros((n, n), dtype=np.complex)

    def inner(self, x, u, v):
        return np.real(
            np.tensordot(la.solve(x, u), multitransp(la.solve(x, v)),
                         axes=x.ndim))

    def norm(self, x, u):
        # This implementation is as fast as np.linalg.solve_triangular and is
        # more stable, as the above solver tends to output non positive
        # definite results.
        c = la.cholesky(x)
        c_inv = la.inv(c)
        return np.real(
            la.norm(multiprod(multiprod(c_inv, u), multihconj(c_inv))))

    def proj(self, X, G):
        return multiherm(G)

    def egrad2rgrad(self, x, u):
        return multiprod(multiprod(x, multiherm(u)), x)

    def ehess2rhess(self, x, egrad, ehess, u):
        egrad = multiherm(egrad)
        hess = multiprod(multiprod(x, multiherm(ehess)), x)
        hess += multiherm(multiprod(multiprod(u, egrad), x))
        return hess

    def exp(self, x, u):
        k = self._k

        d, q = la.eigh(x)
        if k == 1:
            x_sqrt = q@np.diag(np.sqrt(d))@q.conj().T
            x_isqrt = q@np.diag(1/np.sqrt(d))@q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_sqrt = multiprod(multiprod(q, temp), multihconj(q))

            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(1/np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_isqrt = multiprod(multiprod(q, temp), multihconj(q))

        d, q = la.eigh(multiprod(multiprod(x_isqrt, u), x_isqrt))
        if k == 1:
            e = q@np.diag(np.exp(d))@q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.exp(d[i, :]))[np.newaxis, :, :]
            d = temp
            e = multiprod(multiprod(q, d), multihconj(q))

        e = multiprod(multiprod(x_sqrt, e), x_sqrt)
        e = multiherm(e)
        return e

    def retr(self, x, u):
        r = x + u + (1/2)*u@la.solve(x, u)
        return r

    def log(self, x, y):
        k = self._k

        d, q = la.eigh(x)
        if k == 1:
            x_sqrt = q@np.diag(np.sqrt(d))@q.conj().T
            x_isqrt = q@np.diag(1/np.sqrt(d))@q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_sqrt = multiprod(multiprod(q, temp), multihconj(q))

            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(1/np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_isqrt = multiprod(multiprod(q, temp), multihconj(q))

        d, q = la.eigh(multiprod(multiprod(x_isqrt, y), x_isqrt))
        if k == 1:
            log = q@np.diag(np.log(d))@q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.log(d[i, :]))[np.newaxis, :, :]
            d = temp
            log = multiprod(multiprod(q, d), multihconj(q))

        xi = multiprod(multiprod(x_sqrt, log), x_sqrt)
        xi = multiherm(xi)
        return xi

    def transp(self, x1, x2, d):
        E = multihconj(la.solve(multihconj(x1), multihconj(x2)))
        if self._k == 1:
            E = sqrtm(E)
        else:
            for i in range(len(E)):
                E[i, :, :] = sqrtm(E[i, :, :])
        transp_d = multiprod(multiprod(E, d), multihconj(E))
        return transp_d

    def dist(self, x, y):
        c = la.cholesky(x)
        c_inv = la.inv(c)
        logm = multilog(multiprod(multiprod(c_inv, y), multihconj(c_inv)),
                        pos_def=True)
        return np.real(la.norm(logm))


class SpecialHermitianPositiveDefinite(EuclideanEmbeddedSubmanifold):
    """Manifold of (n x n)^k Hermitian positive
    definite matrices with unit determinant
    called 'Special Hermitian positive definite manifold'.
    It is a totally geodesic submanifold of
    the Hermitian positive definite matrices.
    """
    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        self.HPD = HermitianPositiveDefinite(n, k)

        if k == 1:
            name = ("Manifold of special Hermitian positive definite\
                    ({} x {}) matrices").format(n, n)
        else:
            name = "Product manifold of {} special Hermitian positive\
                    definite ({} x {}) matrices".format(k, n, n)
        dimension = int(k * (n*(n+1) - 1))
        super().__init__(name, dimension)

    def rand(self):
        # Generate k HPD matrices.
        x = self.HPD.rand()

        # Normalize them.
        if self._k == 1:
            x = x / (np.real(la.det(x))**(1/self._n))
        else:
            x = x / (np.real(la.det(x))**(1/self._n)).reshape(-1, 1, 1)

        return x

    def randvec(self, x):
        # Generate k matrices.
        k = self._k
        n = self._n
        if k == 1:
            u = rnd.randn(n, n)+1j*rnd.randn(n, n)
        else:
            u = rnd.randn(k, n, n)+1j*rnd.randn(k, n, n)

        # Project them on tangent space.
        u = self.proj(x, u)

        # Unit norm.
        u = u / self.norm(x, u)

        return u

    def zerovec(self, x):
        return self.HPD.zerovec(x)

    def inner(self, x, u, v):
        return self.HPD.inner(x, u, v)

    def norm(self, x, u):
        return self.HPD.norm(x, u)

    def proj(self, x, u):
        n = self._n
        k = self._k

        # Project matrix on tangent space of HPD.
        u = multiherm(u)

        # Project on tangent space of SHPD at x.
        t = np.trace(la.solve(x, u), axis1=-2, axis2=-1)
        if k == 1:
            u = u - (1/n) * np.real(t) * x
        else:
            u = u - (1/n) * np.real(t.reshape(-1, 1, 1)) * x

        return u

    def egrad2rgrad(self, x, u):
        rgrad = multiprod(multiprod(x, u), x)
        rgrad = self.proj(x, rgrad)
        return rgrad

    def exp(self, x, u):
        e = self.HPD.exp(x, u)

        # Normalize them.
        if self._k == 1:
            e = e / np.real(la.det(e))**(1/self._n)
        else:
            e = e / (np.real(la.det(e))**(1/self._n)).reshape(-1, 1, 1)
        return e

    def retr(self, x, u):
        r = self.HPD.retr(x, u)

        # Normalize them.
        if self._k == 1:
            r = r / np.real(la.det(r))**(1/self._n)
        else:
            r = r / (np.real(la.det(r))**(1/self._n)).reshape(-1, 1, 1)
        return r

    def log(self, x, y):
        return self.HPD.log(x, y)

    def transp(self, x1, x2, d):
        return self.proj(x2, self.HPD.transp(x1, x2, d))

    def dist(self, x, y):
        return self.HPD.dist(x, y)


# ---------------------------------------------------------------------
# PRODUCT MANIFOLDS
# ---------------------------------------------------------------------
class WeightedProduct(man.Product):
    """Product manifold with linear combination of metrics."""

    def __init__(self, manifolds, weights=None):
        if weights is None:
            weights = np.ones(len(manifolds))
        self._weights = tuple(weights)
        super().__init__(manifolds)

    @property
    def typicaldist(self):
        raise NotImplementedError

    def inner(self, X, G, H):
        weights = self._weights
        return np.sum([weights[k]*np.squeeze(man.inner(X[k], G[k], H[k]))
                       for k, man in enumerate(self._manifolds)])

    def dist(self, X, Y):
        weights = self._weights
        return np.sqrt(np.sum([weights[k]*np.squeeze(man.dist(X[k], Y[k])**2)
                               for k, man in enumerate(self._manifolds)]))

    def egrad2rgrad(self, X, U):
        weights = self._weights
        return _ProductTangentVector(
            [(1/weights[k])*man.egrad2rgrad(X[k], U[k])
             for k, man in enumerate(self._manifolds)])

    def ehess2rhess(self, X, egrad, ehess, H):
        # Using Koszul formula and R-linearity of affine connections, we get
        # that the Riemannian Hessian of a weighted product manifold
        # is the tuple of the Riemannian Hessians of the different manifolds
        # multiplied by the inverted weights.
        weights = self._weights
        return _ProductTangentVector(
            [(1/weights[k])*man.ehess2rhess(
                X[k], egrad[k], ehess[k], H[k])
             for k, man in enumerate(self._manifolds)])

    def randvec(self, X):
        weights = self._weights
        scale = len(self._manifolds) ** (-1/2)
        return _ProductTangentVector(
            [scale * (1/weights[k]**(-1/2)) * man.randvec(X[k])
             for k, man in enumerate(self._manifolds)])


class KroneckerHermitianPositiveElliptical(WeightedProduct):
    """Manifold of (n x n)^k complex Hermitian positive definite matrices with
    the model: M = A kron B where A and B are complex Hermitian positive
    definite matrices of size (a x a) and (b x b) such that ab = n and
    det(A)=1.


    Attributes
    ----------
    _a : int
        Size of matrix A.
    _b : int
        Size of matrix B.
    _alpha : float
        Coeficient \alpha corresponding to the elliptical model.
    _k : int, optional
        Number of covariance matrices. Default is 1.
    """

    def __init__(self, a, b, alpha, k=1):
        self._n = a*b
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
