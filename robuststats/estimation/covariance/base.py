'''
File: base.py
File Created: Sunday, 20th June 2021 5:00:02 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 21st December 2021 5:04:57 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''
from copy import deepcopy
from joblib import Parallel, delayed
import numpy as np
import numpy.linalg as la
from scipy import linalg
from sklearn.covariance import EmpiricalCovariance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array,\
    _deprecate_positional_args
    
import warnings
import logging

from ...models.mappings import covariancetoreal,\
            covariancetocomplex, arraytoreal, arraytocomplex
from ...utils.checking import check_value, check_positive, check_type

# -----------------------------------------------------------------------
# Normalisation
# -----------------------------------------------------------------------
def get_normalisation_function(normalisation=None):
    """Get normalisation function to perform for shape matrix
    as defined in subsection 3.1 of:
    >Marc Hallin, Davy Paindaveine,
    >Optimal tests for homogeneity of covariance, scale, and shape,
    >Journal of Multivariate Analysis,
    >Volume 100, Issue 3, 2009

    Parameters
    ----------
    normalisaton : string, optional
        type of normalisation to perform.
        Choice between 'trace', 'determinant', 'element'
        and 'None', by default 'None'.

    Returns
    -------
    function
        normalisation function.
    """
    def trace(covariance):
        return np.real(np.trace(covariance)) / covariance.shape[0]

    def determinant(covariance):
        return np.real(la.det(covariance))**(1/covariance.shape[0])

    def element(covariance):
        return covariance[0, 0]

    if normalisation == 'trace':
        return trace
    elif normalisation == 'determinant':
        return determinant
    elif normalisation == 'element':
        return element
    elif normalisation == 'None':
        return (lambda x: 1)
    else:
        logging.error(f'Normalisation type {normalisation} unknown. '
                      'Default to None.')
        return (lambda x: 1)

# -----------------------------------------------------------------------
# Transformer classes
# -----------------------------------------------------------------------
class CovariancesEstimation(BaseEstimator, TransformerMixin):
    """ Estimate several covariances. Inspired by Covariances class
    of pyriemann at:
    >https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/estimation.py
    The difference is that estimator is not a string but an instance
    of a scikit-learn compatible estimator of covariance.
    """
    def __init__(self, estimator, n_jobs=1, **kwargs):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit.
        Do nothing. For compatibility purpose.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_samples, n_features)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.
        Returns
        -------
        self : CovariancesEstimation instance
            The CovariancesEstimation instance.
        """
        return self

    def transform(self, X, **kwargs):
        """Estimate covariance matrices.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_samples, n_features)
            ndarray of trials.
        Returns
        -------
        covmats : ndarray, shape (n_trials, n_features, n_features)
            ndarray of covariance matrices for each trials.
        """
        n_trials, n_samples, n_features = X.shape
        covmats = np.zeros((n_trials, n_features, n_features), dtype=X.dtype)
        if self.n_jobs == 1:
            for i in range(n_trials):
                self.estimator.fit(X[i, :, :], **kwargs)
                covmats[i, :, :] = self.estimator.covariance_
        else:
            cov_list = Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_transform_covariance_copy)(self.estimator, x, **kwargs) 
                    for x in X
                )
            for i, cov in enumerate(cov_list):
                covmats[i, :, :] = cov
        return covmats


def _fit_transform_covariance_copy(estimator, x, **kwargs):
    """Function to do a fit of a scikit-learn estimator and returns the covariance.
    Do a deepcopy of the estimator to avoid serialization errors when used in parallel

    Parameters
    ----------
    estimator : scikit-learn object that inherits EmpiricalCovariance
        the estimator used. The parameters of the fit must already been initialized 
        in the object.
    x : ndarray, shape (n_samples, n_features)
        Data on which perform covariance estimation.

    Returns
    -------
    ndarray, shape (n_features, n_features)
        estimated covariance.
    """
    est = deepcopy(estimator)
    est.fit(x, **kwargs)
    return est.covariance_

# -----------------------------------------------------------------------
# Covariances classes
# -----------------------------------------------------------------------
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
        covariance = np.dot(X.T, X.conj()) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance


class RealEmpiricalCovariance(EmpiricalCovariance, TransformerMixin):
    """Scikit-learn Empirical covariance + transform to get the covariance after.
    """
    def transform(self, X):
        return self.covariance_


class ComplexEmpiricalCovariance(RealEmpiricalCovariance):

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
        """Overridden error_norm of EmpiricalCovariance
        to handle complex data."""

        return np.abs(super().error_norm(comp_cov, norm, scaling,
                                         squared))


# -----------------------------------------------------------------------
# Covariance/Scale classes
# -----------------------------------------------------------------------
class RealCovarianceScale(RealEmpiricalCovariance):
    """Class for estimation of Covariance + scale parameters
    """
    def __init__(self, *, store_precision=False, assume_centered=True):
        super(RealCovarianceScale, self).__init__(
            store_precision=store_precision, assume_centered=assume_centered)
    
    def _set_scale(self, s):
        # set scale
        self.scale_ = check_array(s)
        
    def get_scale(self):
        return self.scale_
    
    def transform(self, X):
        return [self.covariance_, self.scale_]
    
    def fit(self, X):
        raise NotImplementedError("No fit method available.")
        

class ComplexCovarianceScale(ComplexEmpiricalCovariance):
    """Class for estimation of Covariance + scale parameters
    """
    def __init__(self, *, store_precision=False, assume_centered=True):
        super(ComplexCovarianceScale, self).__init__(
            store_precision=store_precision, assume_centered=assume_centered)
    
    def _set_scale(self, s):
        # set scale
        self.scale_ = check_array(s)
        
    def get_scale(self):
        return self.scale_
    
    def transform(self, X):
        return [self.covariance_, self.scale_]
    
    def fit(self, X):
        raise NotImplementedError("No fit method available.")

