'''
File: probability.py
File Created: Monday, 21st June 2021 12:44:26 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 1st July 2021 10:39:31 am
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''
# TODO: Rewrite doc since autodoc can't find scipy doc..

import logging
from scipy.stats._multivariate import  multivariate_normal_gen,\
    multivariate_normal_frozen, multivariate_t_frozen,\
    multivariate_t_gen
import numpy as np
from .mappings import covariancetoreal,\
    arraytoreal, arraytocomplex


def _check_parameters_complex_normal(mean, cov):
    """Check real or complex nature of mutivariate normal
    model parameters and adapt parameters accordingly.

    Parameters
    ----------
    mean : array_like
        Mean of the distribution
    cov : array_like
        Covariance matrix of the distribution

    Returns
    -------
    array_like
        Mean of the distribution, complex if necessary.
    array_like
        Covariance matrix of the distribution, complex if necessary.
    """
    if np.isrealobj(mean) and np.isrealobj(cov):
        return mean, cov
    else:
        return arraytoreal(mean.astype(complex)),\
                covariancetoreal(cov.astype(complex))


def _check_parameters_array_complex_normal(X, mean, cov):
    """Check real or complex nature of mutivariate normal
    model parameters as well as data input and adapt
    parameters accordingly.

    Parameters
    ----------
    X : array_like
        Input data.
    mean : array_like
        Mean of the distribution
    cov : array_like
        Covariance matrix of the distribution

    Returns
    -------
    array_like
        Input data, complex if necessary.
    array_like
        Mean of the distribution, complex if necessary.
    array_like
        Covariance matrix of the distribution, complex if necessary.
    """
    mean_map, cov_map = _check_parameters_complex_normal(mean, cov)
    if np.iscomplexobj(mean):
        return arraytoreal(X.astype(complex)), mean_map, cov_map
    elif np.iscomplexobj(X) and not np.iscomplexobj(mean):
        logging.warning(
            "Input data is complex while parameters of model are not."
            " Discarding imaginary part of data")
        return np.real(X), mean, cov
    else:
        return X, mean, cov


class complex_multivariate_normal_frozen(multivariate_normal_frozen):
    """ Frozen model of complex_multivariate_normal."""

    def __init__(self, mean=None, cov=1, allow_singular=False, seed=None,
                 maxpts=None, abseps=1e-5, releps=1e-5):
        self._mean_orig = mean
        self._cov_orig = cov
        mean, cov = _check_parameters_complex_normal(mean, cov)
        super(complex_multivariate_normal_frozen,
              self).__init__(mean, cov, allow_singular, seed, maxpts,
                             abseps, releps)

    def logpdf(self, x):
        x, _, _ = _check_parameters_array_complex_normal(
            x, self._mean_orig, self._cov_orig)
        return np.real(super(complex_multivariate_normal_frozen,
                             self).logpdf(x))

    def cdf(self, x):
        x, _, _ = _check_parameters_array_complex_normal(
            x, self._mean_orig, self._cov_orig)
        return np.real(super(complex_multivariate_normal_frozen,
                             self).cdf(x))

    def rvs(self, size=1, random_state=None):
        is_complex = np.iscomplexobj(self._mean_orig) or\
                     np.iscomplexobj(self._cov_orig)
        if is_complex:
            return arraytocomplex(super(complex_multivariate_normal_frozen,
                                        self).rvs(size, random_state))
        else:
            return super(complex_multivariate_normal_frozen,
                         self).rvs(size, random_state)


class complex_multivariate_normal_gen(multivariate_normal_gen):
    """ A complex multivariate normal random variable.
    Extends multivariate_normal to circular complex
    distribution thanks to complex to real mappings.
    """

    def __init__(self, seed=None) -> None:
        super(complex_multivariate_normal_gen, self).__init__(seed)

    def __call__(self, mean=None, cov=1, allow_singular=False, seed=None):
        """Create a frozen multivariate normal distribution.
        See `multivariate_normal_frozen` for more information.
        """
        return complex_multivariate_normal_frozen(
            mean, cov,
            allow_singular=allow_singular,
            seed=seed)

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        if np.iscomplexobj(x):
            x = np.asarray(x, dtype=complex)
        else:
            x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        return x

    def logpdf(self, x, mean=None, cov=1, allow_singular=False):
        """Log of the multivariate normal probability density function.
        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`
        Notes
        -----
        %(_mvn_doc_callparams_note)s
        """
        x, mean, cov = _check_parameters_array_complex_normal(
            x, mean, cov)
        return np.real(super(complex_multivariate_normal_gen,
                             self).logpdf(x, mean, cov, allow_singular))

    def pdf(self, x, mean=None, cov=1, allow_singular=False):
        """Multivariate normal probability density function.
        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`
        Notes
        -----
        %(_mvn_doc_callparams_note)s
        """
        x, mean, cov = _check_parameters_array_complex_normal(
            x, mean, cov)
        return np.real(super(complex_multivariate_normal_gen,
                             self).pdf(x, mean, cov, allow_singular))

    def logcdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None,
               abseps=1e-5, releps=1e-5):
        """Log of the multivariate normal cumulative distribution function.
        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
        maxpts : integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps : float, optional
            Absolute error tolerance (default 1e-5)
        releps : float, optional
            Relative error tolerance (default 1e-5)
        Returns
        -------
        cdf : ndarray or scalar
            Log of the cumulative distribution function evaluated at `x`
        Notes
        -----
        %(_mvn_doc_callparams_note)s
        .. versionadded:: 1.0.0
        """
        x, mean, cov = _check_parameters_array_complex_normal(
            x, mean, cov)
        return np.real(super(complex_multivariate_normal_gen,
                             self).logcdf(x, mean, cov, allow_singular,
                                          maxpts, abseps, releps))

    def cdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None,
            abseps=1e-5, releps=1e-5):
        """Multivariate normal cumulative distribution function.
        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
        maxpts : integer, optional
            The maximum number of points to use for integration
            (default `1000000*dim`)
        abseps : float, optional
            Absolute error tolerance (default 1e-5)
        releps : float, optional
            Relative error tolerance (default 1e-5)
        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at `x`
        Notes
        -----
        %(_mvn_doc_callparams_note)s
        .. versionadded:: 1.0.0
        """
        x, mean, cov = _check_parameters_array_complex_normal(
            x, mean, cov)
        return np.real(super(complex_multivariate_normal_gen,
                             self).cdf(x, mean, cov, allow_singular,
                                       maxpts, abseps, releps))

    def rvs(self, mean=None, cov=1, size=1, random_state=None):
        """Draw random samples from a multivariate normal distribution.
        Parameters
        ----------
        %(_mvn_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s
        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.
        Notes
        -----
        %(_mvn_doc_callparams_note)s
        """
        is_complex = np.iscomplexobj(mean) or np.iscomplexobj(cov)
        mean, cov = _check_parameters_complex_normal(mean, cov)
        if is_complex:
            return arraytocomplex(super().rvs(mean, cov, size, random_state))
        else:
            return super(complex_multivariate_normal_gen,
                         self).rvs(mean, cov, size, random_state)

    def entropy(self, mean=None, cov=1):
        """Compute the differential entropy of the multivariate normal.
        Parameters
        ----------
        %(_mvn_doc_default_callparams)s
        Returns
        -------
        h : scalar
            Entropy of the multivariate normal distribution
        Notes
        -----
        %(_mvn_doc_callparams_note)s
        """
        mean, cov = _check_parameters_complex_normal(mean, cov)
        return np.real(super(complex_multivariate_normal_gen,
                       self).entropy(mean, cov))


complex_multivariate_normal = complex_multivariate_normal_gen()


class complex_multivariate_t_frozen(multivariate_t_frozen):
    """ Frozen model of complex_multivariate_t."""

    def __init__(self, loc=None, shape=1, df=1, allow_singular=False,
                 seed=None):
        self._loc_orig = loc
        self._shape_orig = shape
        loc, shape = _check_parameters_complex_normal(loc, shape)
        super(complex_multivariate_t_frozen,
              self).__init__(loc, shape, df, allow_singular, seed)

    def logpdf(self, x):
        x, _, _ = _check_parameters_array_complex_normal(
            x, self._loc_orig, self._shape_orig)
        return np.real(super(complex_multivariate_t_frozen,
                             self).logpdf(x))

    def rvs(self, size=1, random_state=None):
        is_complex = np.iscomplexobj(self._loc_orig) or\
                     np.iscomplexobj(self._shape_orig)
        if is_complex:
            return arraytocomplex(super(
                complex_multivariate_t_frozen, self).rvs(size,
                                                         random_state))
        else:
            return super(complex_multivariate_t_frozen,
                         self).rvs(size, random_state)


class complex_multivariate_t_gen(multivariate_t_gen):
    """ A complex multivariate t-distribution.
    Extends multivariate_t to circular complex
    distribution thanks to complex to real mappings.
    """

    def __init__(self, seed=None) -> None:
        super(complex_multivariate_t_gen, self).__init__(seed)

    def __call__(self, loc=None, shape=1, df=1, allow_singular=False,
                 seed=None):
        """Create a frozen multivariate t-distribution.
        See `multivariate_t_frozen` for more information.
        """
        return complex_multivariate_t_frozen(
            loc, shape, df,
            allow_singular=allow_singular,
            seed=seed)

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        if np.iscomplexobj(x):
            x = np.asarray(x, dtype=complex)
        else:
            x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        return x

    def rvs(self, loc=None, shape=1, df=1, size=1, random_state=None):
        """Draw random samples from a multivariate t-distribution.
        Parameters
        ----------
        %(_mvn_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s
        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.
        Notes
        -----
        %(_mvn_doc_callparams_note)s
        """
        is_complex = np.iscomplexobj(loc) or np.iscomplexobj(shape)
        loc, shape = _check_parameters_complex_normal(loc, shape)
        if is_complex:
            return arraytocomplex(super(complex_multivariate_t_gen,
                                        self).rvs(loc, shape, df, size,
                                                  random_state))
        else:
            return super(complex_multivariate_t_gen,
                         self).rvs(loc, shape, df, size,
                                   random_state)


complex_multivariate_t = complex_multivariate_t_gen()
