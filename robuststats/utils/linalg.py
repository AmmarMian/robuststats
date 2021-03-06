'''
File: linalg.py
File Created: Thursday, 8th July 2021 6:10:40 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Fri Dec 03 2021
Modified By: Ammar Mian
-----
Copyright 2021, Universit√© Savoie Mont-Blanc
'''

import numpy as np
from scipy.linalg import toeplitz
from pymanopt.tools.multi import multitransp
from .backend import get_be_autograd

def multihconj(A, autograd=False):
    be = get_be_autograd(autograd)
    return be.conjugate(multitransp(A))


def multiherm(A, autograd=False):
    # Inspired by MATLAB multiherm function by Nicholas Boumal.
    return 0.5 * (A + multihconj(A, autograd))

# From: https://github.com/alexandrebarachant/pyRiemann/
# blob/master/pyriemann/utils/base.py
def _matrix_operator(Ci, operator, autograd=False):
    """matrix equivalent of an operator."""
    be = get_be_autograd(autograd)
    eigvals, eigvects = be.linalg.eigh(Ci)
    eigvals = be.diag(operator(eigvals))
    Out = be.dot(be.dot(eigvects, eigvals), be.conjugate(eigvects).T)
    return Out


def sqrtm(Ci, autograd=False):
    """Return the matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{1/2}
            \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix square root

    """
    be = get_be_autograd(autograd)
    return _matrix_operator(Ci, be.sqrt, autograd)


def logm(Ci, autograd=False):
    """Return the matrix logarithm of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix logarithm

    """
    be = get_be_autograd(autograd)
    return _matrix_operator(Ci, be.log, autograd)


def expm(Ci, autograd=False):
    """Return the matrix exponential of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix exponential

    """
    be = get_be_autograd(autograd)
    return _matrix_operator(Ci, be.exp, autograd)


def invsqrtm(Ci, autograd=False):
    """Return the inverse matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2}
            \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root

    """
    be = get_be_autograd(autograd)
    def isqrt(x):
        return 1. / be.sqrt(x)
    return _matrix_operator(Ci, isqrt, autograd)


def powm(Ci, alpha, autograd=False):
    """Return the matrix power :math:`\\alpha` of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha}
            \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :param alpha: the power to apply
    :returns: the matrix power

    """
    def power(x):
        return x**alpha
    return _matrix_operator(Ci, power, autograd)


def hermitian(X, autograd=False):
    be = get_be_autograd(autograd)
    return .5*(X + be.conjugate(X.T))


def ToeplitzMatrix(rho, p, dtype=complex):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
                * dtype = dtype of array
            Outputs:
                * the matrix """

    return toeplitz(np.power(rho, np.arange(0, p))).astype(dtype)


def vec(mat):
    return mat.ravel('F')


def vech(mat):
    # Gets Fortran-order
    return mat.T.take(_triu_indices(len(mat)))


def _tril_indices(n):
    rows, cols = np.tril_indices(n)
    return rows * n + cols


def _triu_indices(n):
    rows, cols = np.triu_indices(n)
    return rows * n + cols


def _diag_indices(n):
    rows, cols = np.diag_indices(n)
    return rows * n + cols


def unvec(v):
    k = int(np.sqrt(len(v)))
    assert(k * k == len(v))
    return v.reshape((k, k), order='F')


def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows), dtype=v.dtype)
    result[np.triu_indices(rows)] = v
    result = result + result.conj().T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result