'''
File: _centered_covariance.py
File Created: Tuesday, 21st December 2021 10:02:27 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 21st December 2021 5:46:27 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''


import numpy as np
import numpy.linalg as la
from scipy.stats import chi2, multivariate_t
import logging

from .base import (
    ComplexEmpiricalCovariance,
    RealEmpiricalCovariance,
    get_normalisation_function
)
from ...models.cost import Tyler_cost_real, Tyler_cost_complex
from ...models.probability import complex_multivariate_t
from ._m_estimators import (
    fixed_point_m_estimation_centered, 
    _tyler_m_estimator_function,
    _huber_m_estimator_function,
    _student_t_m_estimator_function
)
from ._natural_gradient import tyler_shape_matrix_naturalgradient
from ...models.mappings import arraytoreal, covariancetocomplex

# -----------------------------------------------------------------------------
# Real-valued estimators
# -----------------------------------------------------------------------------
class CenteredStudentMLE(RealEmpiricalCovariance):
    """Student-t's estimation of Covariance matrix when data is
    centered and the degrees of freedom is known. 
    The approach used is by using a fiexed-point estimator as described
    for example in eq (14) of :
    > Draskovic, Gordana & Pascal, Frederic. (2018). 
    >New Insights Into the Statistical Properties of $M$ -Estimators. 
    >IEEE Transactions on Signal Processing. PP. 10.1109/TSP.2018.2841892. 
    But in real case    
    
    Parameters
    ----------
    df : float
        Degrees of freedom of target multivariate student-t distribution
    tol : float, optional
        criterion for error between two iterations, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations of algorithm, by default 100.
    verbosity : bool, optional
        show a progressbar of algorithm, by default False.
    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    err_ : float
        final error between two iterations.
    iteration_ : int
        number of iterations done for estimation.
    """
    def __init__(self, df, tol=1e-4, 
                 iter_max=100,
                 verbosity=False):
        super().__init__()
        if df<=0:
            raise AttributeError("Degrees of freedom cannot be less than or equal to 0.")
        
        self.df = df
        self.verbosity = verbosity
        self.tol = tol
        self.iter_max = iter_max

        self.err_ = np.inf
        self.iteration_ = 0

    def fit(self, X, y=None, init=None, **kwargs):
        """Fits the Student-t M-estimator of covariance matrix.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        init : array-like of shape (n_features, n_features), optional
            Initial point to start the estimation.
        Returns
        -------
        self : object
        """
        if self.iteration_ > 0 and self.verbosity:
            logging.warning("Overwriting previous fit.")
        X = self._validate_data(X)
        covariance, err, iteration = fixed_point_m_estimation_centered(
            X, _student_t_m_estimator_function, 
            init=init, tol=self.tol,
            iter_max=self.iter_max, verbosity=self.verbosity, 
            df=self.df, n_features=X.shape[1], **kwargs
        )
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None):
        if self.iteration_ ==0:
            self.fit(X_test)
        logpdf = multivariate_t(
            shape=self.covariance_, df=self.df).logpdf(X_test)
        return np.sum(logpdf)


class HuberMEstimator(RealEmpiricalCovariance):
    """Huber's M-estimation of covariance matrix by fixed-point algorithm.
    Data is always assumed to be centered and real-valued.
    
    Estimator is solution of the equation :
    $$
    \widehat{\mathbf{M}}_{\mathrm{Hub}}=
    \frac{1}{N b} \sum_{n=1}^{N} \mathbf{z}_{n} \mathbf{z}_{n}^{\mathrm{T}} 
    \mathbb{1}_{\mathbf{z}_{n}^{\mathrm{T}} \widehat{\mathbf{M}}_{\mathrm{Hub}}^{-1} \mathbf{z}_{n} \leq a}
    +
    \frac{1}{N b} \sum_{n=1}^{N} \frac{\mathbf{z}_{n} \mathbf{z}_{n}^{\mathrm{T}}}{\mathbf{z}_{n}^{\mathrm{T}} \widehat{\mathbf{M}}_{\mathrm{Hub}}^{-1} \mathbf{z}_{n}}
    \mathbb{1}_{\mathbf{z}_{n}^{\mathrm{T}} \widehat{\mathbf{M}}_{\mathrm{Hub}}^{-1} \mathbf{z}_{n} \geq a}.
    $$
    
    For details, see:
    > Contributions aux traitements robustes pour les systèmes multi-capteurs
    > Bruno Meriaux, 2020
    > p. 44, eq (3.14)
    
    Parameters
    ----------
    q : float
        percent of values deemed uncorrupted.
    iter_max : int, optional
        number of maximum iterations of algorithm, by default 100.
    verbosity : bool, optional
        show a progressbar of algorithm, by default False.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    err_ : float
        final error between two iterations.
    iteration_ : int
        number of iterations done for estimation.
    """
    def __init__(self, q, tol=1e-4, 
                 iter_max=100,
                 verbosity=False):
        super().__init__()
        self.q = q
        self.verbosity = verbosity
        self.tol = tol
        self.iter_max = iter_max

        self.err_ = np.inf
        self.iteration_ = 0

    def fit(self, X, y=None, init=None, **kwargs):
        """Fits the Student-t M-estimator of covariance matrix.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        init : array-like of shape (n_features, n_features), optional
            Initial point to start the estimation.
        Returns
        -------
        self : object
        """
        if self.iteration_ > 0 and self.verbosity:
            logging.warning("Overwriting previous fit.")
        X = self._validate_data(X)
        N, p = X.shape
        # Estimating lambda and beta
        lbda = chi2.ppf(self.q, p)/2    
        beta = chi2.cdf(2*lbda, p+1) + 2*lbda*(1-self.q)/p
        covariance, err, iteration = fixed_point_m_estimation_centered(
            X, _huber_m_estimator_function, 
            init=init, tol=self.tol,
            iter_max=self.iter_max, verbosity=self.verbosity, 
            lbda=lbda, beta=beta, **kwargs
        )
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None):
        # TODO : implement score of Huber M-estimator
        raise NotImplementedError("Sorry : score isn't implemented yet")



class TylerShapeMatrix(RealEmpiricalCovariance):
    """Tyler M-estimator of shape matrix with real values.
    See:
    >David E. Tyler.
    >"A Distribution-Free M-Estimator of Multivariate Scatter."
    >Ann. Statist. 15 (1) 234 - 251, March, 1987.
    >https://doi.org/10.1214/aos/1176350263

    Parameters
    ----------
    tol : float, optional
        criterion for error between two iterations, by default 1e-4.
    method : str, optional
        way to compute the solution between
        'fixed-point', 'bcd' or 'gradient', by default 'fixed-point'.
    iter_max : int, optional
        number of maximum iterations of algorithm, by default 100.
    normalisation : str, optional
        type of normalisation between 'trace', 'determinant'
        or 'None', by default 'None'.
    verbosity : bool, optional
        show a progressbar of algorithm, by default False.
    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    err_ : float
        final error between two iterations.
    iteration_ : int
        number of iterations done for estimation.
    """

    def __init__(self, tol=1e-4, method='fixed-point',
                 iter_max=100, normalisation='None',
                 verbosity=False):
        super().__init__()
        self.method = method
        self.verbosity = verbosity
        self.tol = tol
        self.iter_max = iter_max
        self.normalisation = normalisation

        self.set_method()
        self.err_ = np.inf
        self.iteration_ = 0

    def set_method(self):
        # TODO: Add BCD
        if self.method == 'fixed-point':
            def estimation_function(X, init):
                S = get_normalisation_function(self.normalisation)
                n_features = X.shape[1]
                sigma, err, iteration = fixed_point_m_estimation_centered(
                    X, _tyler_m_estimator_function,
                    init=init, tol=self.tol, iter_max=self.iter_max,
                    verbosity=self.verbosity,
                    n_features=n_features
                )
                return sigma/S(sigma), err, iteration

        elif self.method == 'natural gradient':
            def estimation_function(X, init):
                return tyler_shape_matrix_naturalgradient(
                    X, init=init, normalisation=self.normalisation,
                    verbosity=self.verbosity)
        else:
            logging.error("Estimation method not known.")
            raise NotImplementedError(f"Estimation method {self.method}"
                                      " is not known.")

        self._estimation_function = estimation_function

    def fit(self, X, y=None, init=None, **kwargs):
        """Fits the Tyler estimator of shape matrix with the
        specified method when initialised object.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        init : array-like of shape (n_features, n_features), optional
            Initial point to start the estimation.
        Returns
        -------
        self : object
        """
        if self.iteration_ > 0 and self.verbosity:
            logging.warning("Overwriting previous fit.")
        X = self._validate_data(X)
        covariance, err, iteration = self._estimation_function(
            X, init)
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None):
        if self.iteration_ > 0:
            self.fit(X_test)
        return Tyler_cost_real(X_test, self.covariance_)
    
    
# -----------------------------------------------------------------------------
# Complex-valued estimators
# -----------------------------------------------------------------------------
class ComplexCenteredStudentMLE(ComplexEmpiricalCovariance):
    """Student-t's estimation of Covariance matrix when data is
    centered and the degrees of freedom is known. 
    The approach used is by using a fiexed-point estimator as described
    for example in eq (14) of :
    > Draskovic, Gordana & Pascal, Frederic. (2018). 
    >New Insights Into the Statistical Properties of $M$ -Estimators. 
    >IEEE Transactions on Signal Processing. PP. 10.1109/TSP.2018.2841892. 
    

    Since the choice of the division by 2 of the degrees of freedom depends
    upon the choice done in the pdf of Student-t, we chose to be coherent with
    scipy.stats implementation of multivariate t model.
    
    Parameters
    ----------
    df : float
        Degrees of freedom of target multivariate student-t distribution
    tol : float, optional
        criterion for error between two iterations, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations of algorithm, by default 100.
    verbosity : bool, optional
        show a progressbar of algorithm, by default False.
    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    err_ : float
        final error between two iterations.
    iteration_ : int
        number of iterations done for estimation.
    """
    def __init__(self, df, tol=1e-4, 
                 iter_max=100,
                 verbosity=False):
        super().__init__()
        self.df = df
        self.verbosity = verbosity
        self.tol = tol
        self.iter_max = iter_max

        self.err_ = np.inf
        self.iteration_ = 0

    def fit(self, X, y=None, init=None, **kwargs):
        """Fits the Student-t M-estimator of covariance matrix.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        init : array-like of shape (n_features, n_features), optional
            Initial point to start the estimation.
        Returns
        -------
        self : object
        """
        if self.iteration_ > 0 and self.verbosity:
            logging.warning("Overwriting previous fit.")
        X = self._validate_data(X)
        covariance, err, iteration = fixed_point_m_estimation_centered(
            X, _student_t_m_estimator_function, 
            init=init, tol=self.tol,
            iter_max=self.iter_max, verbosity=self.verbosity, 
            df=self.df, n_features=X.shape[1], **kwargs
        )
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None, model='Gaussian'):
        if self.iteration_ ==0:
            logging.error("Estimator hasn't been fitted yet !")
            return None
        logpdf = complex_multivariate_t(
            shape=self.covariance_, df=self.df).logpdf(X_test)
        return np.sum(logpdf)


class ComplexHuberMEstimator(ComplexEmpiricalCovariance):
    """Huber's M-estimation of covariance matrix by fixed-point algorithm.
    Data is always assumed to be centered and complex-valued.
    
    Estimator is solution of the equation :
    $$
    \widehat{\mathbf{M}}_{\mathrm{Hub}}=
    \frac{1}{N b} \sum_{n=1}^{N} \mathbf{z}_{n} \mathbf{z}_{n}^{\mathrm{H}} 
    \mathbb{1}_{\mathbf{z}_{n}^{\mathrm{H}} \widehat{\mathbf{M}}_{\mathrm{Hub}}^{-1} \mathbf{z}_{n} \leq a}
    +
    \frac{1}{N b} \sum_{n=1}^{N} \frac{\mathbf{z}_{n} \mathbf{z}_{n}^{\mathrm{H}}}{\mathbf{z}_{n}^{\mathrm{H}} \widehat{\mathbf{M}}_{\mathrm{Hub}}^{-1} \mathbf{z}_{n}}
    \mathbb{1}_{\mathbf{z}_{n}^{\mathrm{H}} \widehat{\mathbf{M}}_{\mathrm{Hub}}^{-1} \mathbf{z}_{n} \geq a}.
    $$
    
    For details, see:
    > Contributions aux traitements robustes pour les systèmes multi-capteurs
    > Bruno Meriaux, 2020
    > p. 44, eq (3.14)
    
    Parameters
    ----------
    q : float
        percentage of samples deemed uncorrupted.
    iter_max : int, optional
        number of maximum iterations of algorithm, by default 100.
    verbosity : bool, optional
        show a progressbar of algorithm, by default False.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    err_ : float
        final error between two iterations.
    iteration_ : int
        number of iterations done for estimation.
    """
    def __init__(self, q, tol=1e-4, 
                 iter_max=100,
                 verbosity=False):
        super().__init__()
        self.q = q
        self.verbosity = verbosity
        self.tol = tol
        self.iter_max = iter_max

        self.err_ = np.inf
        self.iteration_ = 0


    def fit(self, X, y=None, init=None, **kwargs):
        """Fits the Huber's M-estimator of covariance matrix.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        init : array-like of shape (n_features, n_features), optional
            Initial point to start the estimation.
        Returns
        -------
        self : object
        """
        if self.iteration_ > 0 and self.verbosity:
            logging.warning("Overwriting previous fit.")
        X = self._validate_data(X)
        N, p = X.shape
        # Estimating lambda and beta
        lbda = chi2.ppf(self.q, p)/2    
        beta = chi2.cdf(2*lbda, p+1) + 2*lbda*(1-self.q)/p
        covariance, err, iteration = fixed_point_m_estimation_centered(
            X, _huber_m_estimator_function, 
            init=init, tol=self.tol,
            iter_max=self.iter_max, verbosity=self.verbosity, 
            lbda=lbda, beta=beta, **kwargs
        )
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None):
        # TODO : implement score of Huber M-estimator
        raise NotImplementedError("Sorry : score isn't implemented yet")


class ComplexTylerShapeMatrix(ComplexEmpiricalCovariance):
    """Tyler M-estimator of shape matrix with complex values.
    See:
    >David E. Tyler.
    >"A Distribution-Free M-Estimator of Multivariate Scatter."
    >Ann. Statist. 15 (1) 234 - 251, March, 1987.
    >https://doi.org/10.1214/aos/1176350263

    Parameters
    ----------
    tol : float, optional
        criterion for error between two iterations, by default 1e-4.
    method : str, optional
        way to compute the solution between
        'fixed-point', 'bcd' or 'gradient', by default 'fixed-point'.
    iter_max : int, optional
        number of maximum iterations of algorithm, by default 100.
    normalisation : str, optional
        type of normalisation between 'trace', 'determinant'
        or 'None', by default 'None'.
    verbosity : bool, optional
        show a progressbar of algorithm, by default False.
    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    err_ : float
        final error between two iterations.
    iteration_ : int
        number of iterations done for estimation.
    """

    def __init__(self, tol=1e-4, method='fixed-point',
                 iter_max=100, normalisation='None',
                 verbosity=False):
        super().__init__()
        self.method = method
        self.verbosity = verbosity
        self.tol = tol
        self.iter_max = iter_max
        self.normalisation = normalisation

        self.set_method()
        self.err_ = np.inf
        self.iteration_ = 0

    def set_method(self):
        # TODO: Add BCD
        if self.method == 'fixed-point':
            def estimation_function(X, init):
                S = get_normalisation_function(self.normalisation)
                n_features = X.shape[1]
                sigma, err, iteration = fixed_point_m_estimation_centered(
                    X, _tyler_m_estimator_function,
                    init=init, tol=self.tol, iter_max=self.iter_max,
                    verbosity=self.verbosity,
                    n_features=n_features
                )
                return sigma/S(sigma), err, iteration

        elif self.method == 'natural gradient':
            def estimation_function(X, init):
                
                X_real = arraytoreal(X)
                Sigma, cost_value, _ = tyler_shape_matrix_naturalgradient(
                    X_real, init=init, normalisation='None',
                    verbosity=self.verbosity
                )
                S = get_normalisation_function(self.normalisation)
                Sigma = covariancetocomplex(Sigma)
                Sigma = (Sigma + np.conj(Sigma).T)/2
                return Sigma/S(Sigma), cost_value, None
        else:
            logging.error("Estimation method not known.")
            raise NotImplementedError(f"Estimation method {self.method}"
                                      " is not known.")

        self._estimation_function = estimation_function

    def fit(self, X, y=None, init=None, **kwargs):
        """Fits the Tyler estimator of shape matrix with the
        specified method when initialised object.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        init : array-like of shape (n_features, n_features), optional
            Initial point to start the estimation.
        Returns
        -------
        self : object
        """
        if self.iteration_ > 0 and self.verbosity:
            logging.warning("Overwriting previous fit.")
        X = self._validate_data(X)
        covariance, err, iteration = self._estimation_function(
            X, init)
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None, model='Gaussian'):
        if self.iteration_ > 0:
            self.fit(X_test)
        return Tyler_cost_complex(X_test, self.covariance_)

