'''
File: base.py
File Created: Sunday, 20th June 2021 5:00:02 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 8th July 2021 1:58:59 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''
import numpy as np
from scipy import linalg
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils.validation import check_array,\
    _deprecate_positional_args
import warnings

from ..models.mappings import covariancetoreal,\
            covariancetocomplex, arraytoreal, arraytocomplex


@_deprecate_positional_args
def empirical_covariance(X, *, assume_centered=False):
    """Computes the Maximum likelihood covariance estimator when
    data is complex.
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data will be centered before computation.
    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)
        Empirical covariance (Maximum Likelihood Estimator).
    Examples
    --------
    >>> from sklearn.covariance import empirical_covariance
    >>> X = [[1,1,1],[1,1,1],[1,1,1],
    ...      [0,0,0],[0,0,0],[0,0,0]]
    >>> empirical_covariance(X)
    array([[0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25]])
    """
    X = np.asarray(X, dtype=complex)

    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")

    if assume_centered:
        covariance = np.dot(X.T.conj(), X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance


class ComplexEmpiricalCovariance(EmpiricalCovariance):

    def _set_covariance(self, covariance):
        """Saves the covariance and precision estimates
        Storage is done accordingly to `self.store_precision`.
        Precision stored only if invertible.
        Parameters
        ----------
        covariance : array-like of shape (n_features, n_features)
            Estimated covariance matrix to be stored, and from which precision
            is computed.
        """
        covariance = covariancetocomplex(
            check_array(covariancetoreal(covariance)))
        # set covariance
        self.covariance_ = covariance
        # set precision
        if self.store_precision:
            self.precision_ = linalg.pinvh(covariance, check_finite=False)
        else:
            self.precision_ = None

    def _validate_data(self, X):
        X_verified = arraytocomplex(
         super(ComplexEmpiricalCovariance, self)._validate_data(arraytoreal(X))
        )
        return X_verified

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
            self.location_ = np.zeros(X.shape[1], dtype=complex)
        else:
            self.location_ = X.mean(0)
        covariance = empirical_covariance(
            X, assume_centered=self.assume_centered)

        self._set_covariance(covariance)

        return self

    def error_norm(self, comp_cov, norm='frobenius', scaling=True,
                   squared=True):
        """Overidden error_norm of EmpiricalCovariance
        to handle complex data."""

        return np.abs(super().error_norm(comp_cov, norm, scaling,
                                         squared))
