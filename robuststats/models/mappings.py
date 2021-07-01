'''
File: mappings.py
Created Date: Friday June 18th 2021 - 05:32pm
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Thursday, 1st July 2021 9:52:47 am
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright (c) 2021 Université Savoie Mont-Blanc
'''

import numpy as np
# import numpy.linalg as la
import logging


def arraytoreal(a):
    """Returns a real equivalent of input complex array used in various taks.

    Parameters
    ----------
    a : array-like of shape (n_samples, n_features)
        Input array.

    Returns
    -------
    array-like of shape (n_samples, 2*n_features)
        Real equivalent array.

    Raises
    ------
    AttributeError
        when input array format is not of dimension 1 or 2.
    """
    if np.iscomplexobj(a):
        if a.ndim == 1:
            return np.concatenate([np.real(a), np.imag(a)])
        elif a.ndim == 2:
            return np.hstack([np.real(a), np.imag(a)])
        else:
            raise AttributeError("Input array format not supported.")
    else:
        logging.debug("Input array is not complex, returning input")
        return a


def arraytocomplex(a):
    """Returns complex array from real input array.

    Parameters
    ----------
    a : array-like of shape (n_samples, 2*n_features)
        Input array.

    Returns
    -------
    array-like of shape (n_samples, 2*n_features)
        Real equivalent array.

    Raises
    ------
    AttributeError
        when input array format is not of dimension 1 or 2.
    """
    if not np.iscomplexobj(a):
        if a.ndim == 1:
            p = int(len(a)/2)
            return a[:p] + 1j*a[p:]
        elif a.ndim == 2:
            p = int(a.shape[1]/2)
            return np.vstack(a[:, :p] + 1j*a[:, p:])
        else:
            raise AttributeError("Input array format not supported")
    else:
        return a


def covariancestoreal(a):
    """Same as :func:`~robuststats.models.mappings.complexreal.covariancetoreal
    but apply it to several matrices in input.

    Parameters
    ----------
    a : array-like of shape (n_samples, n_features, n_features)
        Input matrices along axis 1 and 2.

    Returns
    -------
    array-like of shape (n_samples, 2*n_features, 2*n_features)
        Real equivalent to matrices in input.

    Raises
    ------
    AttributeError
        when input array format is not of dimension 3.
    """
    if not np.iscomplexobj(a):
        logging.debug("Input array is not complex, returning input")
        return a
    elif a.ndim == 3:
        n_samples, n_features, _ = a.shape
        a_real = np.zeros((n_samples, 2*n_features, 2*n_features))
        for i in range(n_samples):
            a_real[i] = covariancetoreal(a[i])
        return a_real
    else:
        raise AttributeError("Input array format not supported.")


def covariancestocomplex(a):
    """Same as :func:`~robuststats.models.mappings.complexreal.covariancetocomplex
    but apply it to several matrices in input.

    Parameters
    ----------
    a : array-like of shape (n_samples, 2*n_features, 2*n_features)
        Input matrices along axis 1 and 2.

    Returns
    -------
    array-like of shape (n_samples, n_features, n_features)
        Complex matrices from real matrices in input.

    Raises
    ------
    AttributeError
        when input array format is not of dimension 2 or shape is not even.
    """
    if np.iscomplexobj(a):
        logging.debug("Input array is already complex, returning input.")
        return a
    elif a.ndim == 3 and a.shape[1] % 2 == 0:
        n_samples, n_features, _ = a.shape
        a_complex = np.zeros(
            (n_samples, int(n_features/2), int(n_features/2)), dtype=complex)
        for i in range(n_samples):
            a_complex[i] = covariancetocomplex(a[i])
        return a_complex
    else:
        raise AttributeError("Input array format not supported.")


def covariancetoreal(a):
    """Return real equivalent of complex matrix input.

    Parameters
    ----------
    a : array-like of shape (n_features, n_features)
        Input array.

    Returns
    -------
    array-like of shape (2*n_features, 2*n_features)
        Real equivalent of input array.

    Raises
    ------
    AttributeError
        when input array is not a covariance matrix.
    """

    if np.iscomplexobj(a):
        if iscovariance(a):
            real_matrix = .5 * np.block([[np.real(a), -np.imag(a)],
                                        [np.imag(a), np.real(a)]])
            return real_matrix
        else:
            raise AttributeError("Input array is not a covariance.")
    else:
        logging.debug("Input array is not complex, returning input.")
        return a


def covariancetocomplex(a):
    """Return complex matrix from its real equivalent in input.
    Input can be any transform of a matrix obtained thanks to function
    covariancetoreal or any square amtrix whose shape is an even number.

    Parameters
    ----------
    a : array-like of shape (2*n_features, 2*n_features)
        Input array, real equivalent of a complex square matrix.

    Returns
    -------
    array-like of shape (n_features, n_features)
        Real equivalent of input array.

    Raises
    ------
    AttributeError
        when input array format is not of dimension 2 or shape is not even.
    """

    if not np.iscomplexobj(a):
        if iscovariance(a) and len(a) % 2 == 0:
            p = int(len(a)/2)
            complex_matrix = 2 * a[:p, :p] + 2j*a[p:, :p]
            return complex_matrix
        else:
            raise AttributeError("Input array format not supported.")

    else:
        logging.debug("Input is already a complex array, returning input.")
        return a


def iscovariance(a):
    """Check if Input array correspond to a square matrix.
    TODO: do more than square matrix.

    Parameters
    ----------
    a : array-like
        Input array to check.

    Returns
    -------
    bool
        Return True if the input array is a square matrix.
    """

    return (a.ndim == 2) and (a.shape[0] == a.shape[1])


def check_Hermitian(a, rtol=1e-05, atol=1e-08):
    """Check wheter the matrix a in input is a Hermitian matrix to
    a given precision.

    Parameters
    ----------
    a : array-like
        Input array
    rtol : float, optional
        The relative tolerance parameter, by default 1e-05.
    atol : float, optional
        The absolute tolerance parameter, by default 1e-08.

    Returns
    -------
    bool
        Returns True if the input array is Hermitian up to the given precision.
    """

    return np.allclose(a, np.conj(a.T), rtol=rtol, atol=atol)
