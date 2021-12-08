'''
File: base.py
File Created: Sunday, 20th June 2021 5:00:02 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Wednesday, 8th December 2021 4:08:00 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''
from copy import deepcopy
from joblib import Parallel, delayed
import numpy as np
from scipy import linalg
from sklearn.covariance import EmpiricalCovariance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array,\
    _deprecate_positional_args
import warnings
from ..models.mappings import covariancetoreal,\
            covariancetocomplex, arraytoreal, arraytocomplex
from ..utils.checking import check_value, check_positive, check_type

# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------
class _FeatureArray():
    """Class allowing to store data belonging on a manifold.
    """
    def __init__(self, *shape):
        self._array = None
        self._shape = shape
        self._size_preallocation = int(1e3)
        self._len = 0
        self._tol = 1e-5

    def __str__(self):
        return self._array.__str__()
    
    def __repr__(self):
        return self._array.__str__()

    def __empty(self):
        return len(self) == 0

    def __len__(self):
        return self._len

    @property
    def dtype(self):
        if self.__empty():
            return tuple()
        return tuple([self._array[i].dtype for i in range(len(self._array))])

    @property
    def shape(self):
        if self.__empty():
            return self.__len__()
        shape = list()
        for i in range(len(self._shape)):
            shape.append((len(self), *(self._shape[i])))
        shape = tuple(shape)
        return shape

    @property
    def nb_manifolds(self):
        return len(self._shape)

    def __getitem__(self, key):
        check_positive(len(self), 'self _FeatureArray', strictly=True)

        a = self._array
        a = [a[i][:len(self)] for i in range(len(a))]
        temp = [a[i][key] for i in range(len(a))]
        if type(key) == int:
            temp = [temp[i][np.newaxis, ...] for i in range(len(temp))]
        f_a = _FeatureArray(*[temp[i].shape[1:] for i in range(len(temp))])
        f_a.append(temp)
        return f_a

    def append(self, data):
        check_type(data, 'data', [np.ndarray, list, tuple, _FeatureArray])

        if type(data) == _FeatureArray:
            data = data.export()

        if type(data) == np.ndarray:
            data = [data]

        if self._array is None:
            self._array = [None]*len(self._shape)

        check_value(self.nb_manifolds, 'self.nb_manifolds', [len(data)])

        for i, (a, d) in enumerate(zip(self._array, data)):
            check_type(d, 'd', [np.ndarray, np.memmap])
            if a is not None:
                check_value(d.dtype, 'd.dtype', [a.dtype])

            # Add batch dim.
            if len(d.shape) == len(self._shape[i]):
                d = d[np.newaxis, ...]

            check_value(d.ndim, 'd.ndim', [len(self._shape[i])+1])

            if a is None:
                self._array[i] = d
            else:
                shape = (self._size_preallocation, *(self._shape[i]))
                while len(self) + len(d) > len(self._array[i]):
                    a = self._array[i]
                    temp = np.zeros(shape, dtype=a.dtype)
                    self._array[i] = np.concatenate([a, temp], axis=0)
                self._array[i][len(self):len(self)+len(d)] = d

        self._len += len(d)

    def vectorize(self):
        check_positive(len(self), 'len(self)', strictly=True)

        temp = list()
        for a in self._array:
            vec = None
            bs = len(a)

            # check if matrix
            if a.ndim == 3:
                # check if square
                _, p, q = a.shape
                if p == q:
                    # check if matrices are symmetric or skew-symmetric
                    a_H = np.transpose(a, axes=(0, 2, 1)).conj()
                    condition_sym = (np.abs(a - a_H) < self._tol).all()
                    condition_skew = (np.abs(a + a_H) < self._tol).all()
                    if condition_sym or condition_skew:
                        indices = np.triu_indices(p)
                        a = a[:, indices[0], indices[1]]

            vec = a.reshape((bs, -1))

            temp.append(vec)

        vec = np.concatenate(temp, axis=1)

        return vec

    def export(self):
        a = [self._array[i][:len(self)] for i in range(self.nb_manifolds)]
        for i in range(len(a)):
            if len(a[i]) == 1:
                a[i] = np.squeeze(a[i], axis=0)
        if self.nb_manifolds == 1:
            a = a[0]
        return a
    

def _feature_estimation(estimator):
    def wrapper(data, **kwargs):
        # estimation
        f = deepcopy(estimator)
        f = estimator.fit_transform(data, **kwargs)

        # return a _FeatureArray
        if type(f) in [np.float64, np.complex128]:
            f = np.array([f])
        if type(f) == np.ndarray:
            f = [f]
        f_a = _FeatureArray(*[f[i].shape for i in range(len(f))])
        f_a.append(f)

        return f_a
    return wrapper

# -----------------------------------------------------------------------
# Transformer classes
# -----------------------------------------------------------------------
class FeaturesEstimation(BaseEstimator, TransformerMixin):
    """ Estimate several Features. Inspired by Covariances class
    of pyriemann at:
    >https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/estimation.py
    """
    def __init__(self, estimator, n_jobs=1, **kwargs):
        self.estimator = estimator
        self.n_jobs = n_jobs

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
        features : _FeatureArray, corresponding to the features for each trials.
        """
        
        estimation_function = _feature_estimation(self.estimator)
        features_list = Parallel(n_jobs=self.n_jobs)(
            delayed(estimation_function)(x) for x in X)
        array = features_list[0]
        for data in features_list[1:]:
            array.append(data)
        return array


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
            covmats[i, :, :] = self.estimator.fit_transform(X[i, :, :])
        return covmats


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
        covariance = np.dot(X.T.conj(), X) / X.shape[0]
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
        """Overidden error_norm of EmpiricalCovariance
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
        