'''
File: base.py
File Created: Sunday, 20th June 2021 5:00:02 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 28th October 2021 10:29:52 am
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''
import numpy as np
from scipy import linalg
from sklearn.covariance import EmpiricalCovariance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array,\
    _deprecate_positional_args
import warnings

from ..models.mappings import covariancetoreal,\
            covariancetocomplex, arraytoreal, arraytocomplex


class CovariancesEstimation(BaseEstimator, TransformerMixin):
    """ Estimate several covariances. Inspired by Covariances class
    of pyriemann at:
    >https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/estimation.py
    The difference is that estimator is not a string but an instance
    of a scikit-learn compatible estimator of covariance.
    """
    def __init__(self, estimator, **kwargs):
        self.estimator = estimator

    def fit(self, X, y=None):
        """Fit.
        Do nothing. For compatibility purpose.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.
        Returns
        -------
        self : Covariances instance
            The Covariances instance.
        """
        return self

    def transform(self, X):
        """Estimate covariance matrices.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of covariance matrices for each trials.
        """
        Nt, Ne, Ns = X.shape
        covmats = np.zeros((Nt, Ne, Ne), dtype=X.dtype)
        for i in range(Nt):
            covmats[i, :, :] = self.estimator.fit(X[i, :, :])
        return covmats


@_deprecate_positional_args
def complex_empirical_covariance(X, *, assume_centered=False):
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
        covariance = complex_empirical_covariance(
            X, assume_centered=self.assume_centered)

        self._set_covariance(covariance)

        return self

    def error_norm(self, comp_cov, norm='frobenius', scaling=True,
                   squared=True):
        """Overidden error_norm of EmpiricalCovariance
        to handle complex data."""

        return np.abs(super().error_norm(comp_cov, norm, scaling,
                                         squared))
