'''
File: covariance.py
File Created: Sunday, 20th June 2021 8:38:42 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 21st December 2021 4:46:18 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import numpy as np
import numpy.linalg as la
import logging

from scipy.stats import multivariate_t
from scipy.stats import chi2

from sklearn.utils.validation import _deprecate_positional_args
from sklearn.covariance import EmpiricalCovariance, empirical_covariance
from .covariance.base import complex_empirical_covariance, ComplexEmpiricalCovariance,\
                RealEmpiricalCovariance
from ..models.mappings import arraytoreal, arraytocomplex, covariancetocomplex
from ..models.manifolds import HermitianPositiveDefinite
from ..models.cost import Tyler_cost_real, Tyler_cost_complex
from ..models.probability import complex_multivariate_t

from ..utils.verbose import logging_tqdm
from ..utils.linalg import invsqrtm
from tqdm import tqdm

from pymanopt.function import Callable
from pymanopt import Problem
from pymanopt.manifolds.psd import SymmetricPositiveDefinite
from pymanopt.solvers import SteepestDescent
import autograd
import autograd.numpy as np_a
import autograd.numpy.linalg as a_la

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


# -----------------------------------------------------------------------------
# Real-valued estimators
# -----------------------------------------------------------------------------
@_deprecate_positional_args
def _huber_m_estimator_function(x, lbda, beta):
    """Huber M-estimator function as defined for example in
    > Statistiques des estimateurs robustes pour le traitement du signal et des images
    > p 16, Ph.d Thesis, Gordana Draskovic
    
    It consists of the function defined for a threshold $\lambda$ and a real $\beta$
    by the equation :
    $$
    u(x)=\frac{1}{\beta}\min\left(1, \lambda/x\right)
    $$

    Parameters
    ----------
    x : array-like
        input data
    lbda : float
        value at which we shift from no ponderation to inverse ponderation
    beta : float
        a tuning parameter

    Returns
    -------
    array-like
        array of the same shape as input with values that have been scaled depending
        on the tuning parameters

    """
    if lbda<=0 or beta<0:
        raise AssertionError(
            f"Error, lambda and beta can't be negative : lambda={lbda}, beta={beta}"
        )
    return (1/beta)*np.minimum(1,lbda/x)

@_deprecate_positional_args
def huber_m_estimator_fixed_point(X, q, init=None, tol=1e-4,
                                iter_max=30, verbosity=False, **kwargs):
    """Perform Huber's M-estimation of covariance matrix by fixed-point algorithm.
    Data is always assumed to be centered and real-valued.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    q : float
        percentage of samples deemed uncorrupted.
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    tol : float, optional
        tolerance for convergence of algorithm, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations, by default 30.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate.
    float
        final error between two iterations.
    int
        number of iterations done.
    """
    
    # Initialisation
    N, p = X.shape
    

    if init is None:
        sigma = empirical_covariance(X, assume_centered=True)
    else:
        sigma = init
        
    # Estimating lambda and beta
    lbda = chi2.ppf(q, p)/2    
    beta = chi2.cdf(2*lbda, p+1) + 2*lbda*(1-q)/p

    delta = np.inf  # Distance between two iterations
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)

    while (delta > tol) and (iteration < iter_max):
        # compute expression of Tyler estimator
        temp = invsqrtm(sigma)@X.T
        tau = np.einsum('ij,ji->i', temp.T, temp)
        tau = _huber_m_estimator_function(np.real(tau), lbda, beta)
        temp = X.T / np.sqrt(tau)
        sigma_new = (1/N) * temp@temp.T

        # condition for stopping
        delta = la.norm(sigma_new - sigma) / la.norm(sigma)
        iteration += 1

        # updating sigma
        sigma = sigma_new

        if verbosity:
            pbar_v.update()
            pbar_v.set_description(f'(err={delta})', refresh=True)

    if verbosity:
        pbar_v.close()


    if iteration == iter_max:
        logging.warning('Estimation algorithm did not converge')

    return sigma, delta, iteration


@_deprecate_positional_args
def student_t_mle_fixed_point(X, df, init=None, tol=1e-4,
                                iter_max=30, verbosity=False, **kwargs):
    """Perform Student-t's fixed-point estimation of covariance matrix.
    Data is always assumed to be centered and real-valued.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    df : float
        Degrees of freedom of target multivariate student-t distribution
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    tol : float, optional
        tolerance for convergence of algorithm, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations, by default 30.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate.
    float
        final error between two iterations.
    int
        number of iterations done.
    """
    # Initialisation
    N, p = X.shape
    
    if df<=0:
        raise AttributeError("Degrees of freedom cannot be less than or equal to 0.")
    
    if init is None:
        sigma = empirical_covariance(X, assume_centered=True)
    else:
        sigma = init
    delta = np.inf  # Distance between two iterations
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)

    while (delta > tol) and (iteration < iter_max):
        # compute expression of Tyler estimator
        temp = invsqrtm(sigma)@X.T
        tau = np.einsum('ij,ji->i', temp.T, temp)
        tau = df/2 + np.real(tau)
        temp = X.T / np.sqrt(tau)
        sigma_new = ((df/2+p)/N) * temp@temp.T

        # condition for stopping
        delta = la.norm(sigma_new - sigma) / la.norm(sigma)
        iteration += 1

        # updating sigma
        sigma = sigma_new

        if verbosity:
            pbar_v.update()
            pbar_v.set_description(f'(err={delta})', refresh=True)

    if verbosity:
        pbar_v.close()


    if iteration == iter_max:
        logging.warning('Estimation algorithm did not converge')

    return sigma, delta, iteration

@_deprecate_positional_args
def _generate_realTyler_cost_function(X):
    """Generate cost function for gradient descent for Tyler cost function
    as given in eq. (25) of:
    >Wiesel, A. (2012). Geodesic convexity and covariance estimation.
    >IEEE transactions on signal processing, 60(12), 6182-6189.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.

    Returns
    -------
    callable
        function to compute the cost at given data by X
    """

    @Callable
    def cost(Q):
        return Tyler_cost_real(X, Q, autograd=True)
    return cost

@_deprecate_positional_args
def _generate_Tyler_egrad(X, cost):
    """Generate euclidean gradient corresponding to Tyler cost function.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.
    cost :
        cost_function depending on the data X and obtained from
        _generate_realTyler_cost_function.

    Returns
    -------
    callable
        function to compute the cost at given data by X
    """

    n, p = X.shape

    @Callable
    def egrad(Q):
        res = autograd.grad(cost, argnum=[0])(Q)
        return res[0]

    return egrad


@_deprecate_positional_args
def tyler_shape_matrix_naturalgradient(X, init=None, normalisation='None',
                                  verbosity=False, **kwargs):
    """Perform Tyler's estimation of shape matrix with natural 
    gradient on SPD manifold using pymanopt.
    Data is always assumed to be centered.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    normalisaton : string, optional
        type of normalisation to perform.
        Choice between 'trace', 'determinant', 'element'
        and 'None', by default 'None'.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate.
    float
        final error between two iterations.
    int
        Always None. For compatibility purposes
    """
    n_samples, n_features = X.shape
    S = get_normalisation_function(normalisation)
    
    # Normalisation to go to CAE model
    norm_X = la.norm(X, axis=1)
    Y = X / np.tile(norm_X.reshape(n_samples, 1), [1, n_features])

    # Pymanopt setting
    logging.info("Seeting up pymanopt for natural gradient descent")
    manifold = SymmetricPositiveDefinite(n_features)
    cost = _generate_realTyler_cost_function(Y)
    egrad = _generate_Tyler_egrad(Y, cost)
    verbose = verbosity*(verbosity+1)
    problem = Problem(manifold=manifold, cost=cost,
                      egrad=egrad, verbosity=verbose)
    solver = SteepestDescent()
    sigma = solver.solve(problem, x=init)
    return sigma/S(sigma), -cost(sigma), None


@_deprecate_positional_args
def tyler_shape_matrix_fixedpoint(X, init=None, tol=1e-4,
                                  iter_max=30, normalisation='None',
                                  verbosity=False, **kwargs):
    """Perform Tyler's fixed-point estimation of shape matrix.
    Data is always assumed to be centered.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    tol : float, optional
        tolerance for convergence of algorithm, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations, by default 30.
    normalisaton : string, optional
        type of normalisation to perform.
        Choice between 'trace', 'determinant', 'element'
        and 'None', by default 'None'.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate.
    float
        final error between two iterations.
    int
        number of iterations done.
    """

    # Initialisation
    N, p = X.shape
    if init is None:
        sigma = empirical_covariance(X, assume_centered=True)
    else:
        sigma = init

    S = get_normalisation_function(normalisation)
    
    delta = np.inf  # Distance between two iterations
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)

    while (delta > tol) and (iteration < iter_max):
        # compute expression of Tyler estimator
        temp = invsqrtm(sigma)@X.T
        tau = np.einsum('ij,ji->i', temp.T, temp)
        tau = (1/p) * tau
        temp = X.T / np.sqrt(tau)
        sigma_new = (1/N) * temp@temp.T
    
        # condition for stopping
        delta = la.norm(sigma_new - sigma) / la.norm(sigma)
        iteration += 1

        # updating sigma
        sigma = sigma_new

        if verbosity:
            pbar_v.update()
            pbar_v.set_description(f'(err={delta})', refresh=True)

    if verbosity:
        pbar_v.close()

    if iteration == iter_max:
        logging.warning('Estimation algorithm did not converge')

    return sigma/S(sigma), delta, iteration


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
        covariance, err, iteration = huber_m_estimator_fixed_point(
            X, self.q, init=init, tol=self.tol,
            iter_max=self.iter_max, verbosity=self.verbosity, **kwargs
        )
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None):
        # TODO : implement score of Huber M-estimator
        raise NotImplementedError("Sorry : score isn't implemented yet")


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
        covariance, err, iteration = student_t_mle_fixed_point(
            X, self.df, init, self.tol, self.iter_max, self.verbosity
        )
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None):
        if self.iteration_ ==0:
            logging.error("Estimator hasn't been fitted yet !")
            return None
        logpdf = multivariate_t(
            shape=self.covariance_, df=self.df).logpdf(X_test)
        return np.sum(logpdf)


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
            def estimation_function(X, init, **kwargs):
                return tyler_shape_matrix_fixedpoint(
                    X, init=init, tol=self.tol, iter_max=self.iter_max,
                    normalisation=self.normalisation,
                    verbosity=self.verbosity, **kwargs)
        elif self.method == 'natural gradient':
            def estimation_function(X, init, **kwargs):
                return tyler_shape_matrix_naturalgradient(
                    X, init=init, normalisation=self.normalisation,
                    verbosity=self.verbosity, **kwargs)
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
            X, init, **kwargs)
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None, model='Gaussian'):
        # TODO: log-likelihood and all for elliptical distributions
        if model == 'Gaussian':
            return super().score(X_test)
        else:
            raise NotImplementedError(
                f"Model {model} is not known.")


# -----------------------------------------------------------------------------
# Complex-valued estimators
# -----------------------------------------------------------------------------
@_deprecate_positional_args
def complex_huber_m_estimator_fixed_point(X, q, init=None, tol=1e-4,
                                iter_max=30, verbosity=False, **kwargs):
    """Perform Huber's M-estimation of covariance matrix by fixed-point algorithm.
    Data is always assumed to be centered and complex-valued.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    q : float
        percentage of samples deemed uncorrupted.
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    tol : float, optional
        tolerance for convergence of algorithm, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations, by default 30.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate.
    float
        final error between two iterations.
    int
        number of iterations done.
    """
    
    # Initialisation
    N, p = X.shape
    

    if init is None:
        sigma = complex_empirical_covariance(X, assume_centered=True)
    else:
        sigma = init
        
    # Estimating lambda and beta
    lbda = chi2.ppf(q, 2*p)/2    
    beta = chi2.cdf(2*lbda, 2*p+2) + lbda*(1-q)/p    
    
    delta = np.inf  # Distance between two iterations
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)

    while (delta > tol) and (iteration < iter_max):
        # compute expression of Tyler estimator
        temp = invsqrtm(sigma)@X.T
        tau = np.einsum('ij,ji->i', temp.conj().T, temp)
        tau = _huber_m_estimator_function(np.real(tau), lbda, beta)
        temp = X.T / np.sqrt(tau)
        sigma_new = (1/N) * temp@temp.conj().T

        # mask_scm = tau<=a
        # mask_tyler = tau>a
        
        # print(f"SCM: {mask_scm.sum()}")
        # if mask_scm.sum() >= 1:
        #     sigma_new = complex_empirical_covariance(
        #         X[mask_scm], assume_centered=True
        #         ) / b
        # else:
        #     sigma_new = 0
        # temp = X[mask_tyler].T / np.sqrt(tau[mask_tyler])
        # sigma_new += (1/N) * temp@temp.conj().T

        # condition for stopping
        delta = la.norm(sigma_new - sigma) / la.norm(sigma)
        iteration += 1

        # updating sigma
        sigma = sigma_new

        if verbosity:
            pbar_v.update()
            pbar_v.set_description(f'(err={delta})', refresh=True)

    if verbosity:
        pbar_v.close()


    if iteration == iter_max:
        logging.warning('Estimation algorithm did not converge')

    return sigma, delta, iteration


@_deprecate_positional_args
def complex_student_t_mle_fixed_point(X, df, init=None, tol=1e-4,
                                iter_max=30, verbosity=False, **kwargs):
    """Perform Student-t's fixed-point estimation of covariance matrix.
    Data is always assumed to be centered and complex.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    df : float
        Degrees of freedom of target multivariate student-t distribution
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    tol : float, optional
        tolerance for convergence of algorithm, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations, by default 30.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate.
    float
        final error between two iterations.
    int
        number of iterations done.
    """
    # Initialisation
    N, p = X.shape
    
    if df<=0:
        raise AttributeError("Degrees of freedom cannot be less than or equal to 0.")
    
    if init is None:
        sigma = complex_empirical_covariance(X, assume_centered=True)
    else:
        sigma = init
    delta = np.inf  # Distance between two iterations
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)

    while (delta > tol) and (iteration < iter_max):
        # compute expression of Tyler estimator
        temp = invsqrtm(sigma)@X.T
        tau = np.einsum('ij,ji->i', temp.conj().T, temp)
        tau = df + np.real(tau)
        temp = X.T / np.sqrt(tau)
        sigma_new = ((df+p)/N) * temp@temp.conj().T

        # condition for stopping
        delta = la.norm(sigma_new - sigma) / la.norm(sigma)
        iteration += 1

        # updating sigma
        sigma = sigma_new

        if verbosity:
            pbar_v.update()
            pbar_v.set_description(f'(err={delta})', refresh=True)

    if verbosity:
        pbar_v.close()


    if iteration == iter_max:
        logging.warning('Estimation algorithm did not converge')

    return sigma, delta, iteration


def _generate_complexTyler_cost_function(X):
    """Complex version of _generate_realTyler_cost_function
    """
    @Callable
    def cost(Q):
        return Tyler_cost_complex(X, Q, autograd=True)
    return cost


@_deprecate_positional_args
def complex_tyler_shape_matrix_naturalgradient(X, init=None, normalisation='None',
                                  verbosity=False, **kwargs):
    """Perform Tyler's estimation of shape matrix with natural 
    gradient on HPD manifold using pymanopt.
    Data is always assumed to be centered.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    normalisaton : string, optional
        type of normalisation to perform.
        Choice between 'trace', 'determinant', 'element'
        and 'None', by default 'None'.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate.
    float
        final error between two iterations.
    int
        Always None. For compatibility purposes
    """
    X_real = arraytoreal(X)
    Sigma, cost_value, _ = tyler_shape_matrix_naturalgradient(
        X_real, init=init, normalisation='None',
        verbosity=verbosity, **kwargs
    )
    S = get_normalisation_function(normalisation)
    Sigma = covariancetocomplex(Sigma)
    Sigma = (Sigma + np.conj(Sigma).T)/2
    return Sigma/S(Sigma), cost_value, None
    # TODO :Debug why this doesn't work
    # n_samples, n_features = X.shape
    # S = get_normalisation_function(normalisation)

    # # Normalisation to go to CAE model
    # norm_X = la.norm(X, axis=1)
    # Y = X / np.tile(norm_X.reshape(n_samples, 1), [1, n_features])

    # # Pymanopt setting
    # logging.info("Seeting up pymanopt for natural gradient descent")
    # manifold = HermitianPositiveDefinite(n_features)
    # cost = _generate_complexTyler_cost_function(Y)
    # egrad = _generate_Tyler_egrad(Y, cost)
    # verbose = verbosity*(verbosity+1)
    # problem = Problem(manifold=manifold, cost=cost,
    #                   egrad=egrad, verbosity=verbose)
    # solver = SteepestDescent()
    # sigma = solver.solve(problem, x=init)
    # return sigma/S(sigma), -cost(sigma), None


@_deprecate_positional_args
def complex_tyler_shape_matrix_fixedpoint(X, init=None, tol=1e-4,
                                  iter_max=30, normalisation='None',
                                  verbosity=False, **kwargs):
    """Perform Tyler's fixed-point estimation of shape matrix.
    Data is always assumed to be centered and complex.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    tol : float, optional
        tolerance for convergence of algorithm, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations, by default 30.
    normalisaton : string, optional
        type of normalisation to perform.
        Choice between 'trace', 'determinant', 'element'
        and 'None', by default 'None'.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    Returns
    -------
    array-like of shape (n_features, n_features)
        estimate.
    float
        final error between two iterations.
    int
        number of iterations done.
    """

    # Initialisation
    N, p = X.shape
    if init is None:
        sigma = complex_empirical_covariance(X, assume_centered=True)
    else:
        sigma = init

    S = get_normalisation_function(normalisation)
    
    delta = np.inf  # Distance between two iterations
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)

    while (delta > tol) and (iteration < iter_max):
        # compute expression of Tyler estimator
        temp = invsqrtm(sigma)@X.T
        tau = np.einsum('ij,ji->i', temp.conj().T, temp)
        tau = (1/p) * np.real(tau)
        temp = X.T / np.sqrt(tau)
        sigma_new = (1/N) * temp@temp.conj().T
        # condition for stopping
        delta = la.norm(sigma_new - sigma) / la.norm(sigma)
        iteration += 1

        # updating sigma
        sigma = sigma_new

        if verbosity:
            pbar_v.update()
            pbar_v.set_description(f'(err={delta})', refresh=True)


    if verbosity:
        pbar_v.close()


    if iteration == iter_max:
        logging.warning('Estimation algorithm did not converge')

    return sigma/S(sigma), delta, iteration


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
        covariance, err, iteration = complex_huber_m_estimator_fixed_point(
            X, self.q, init=init, tol=self.tol,
            iter_max=self.iter_max, verbosity=self.verbosity, **kwargs
        )
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None):
        # TODO : implement score of Huber M-estimator
        raise NotImplementedError("Sorry : score isn't implemented yet")


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
        covariance, err, iteration = complex_student_t_mle_fixed_point(
            X, self.df, init, self.tol, self.iter_max, self.verbosity
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
            def estimation_function(X, init, **kwargs):
                return complex_tyler_shape_matrix_fixedpoint(
                    X, init=init, tol=self.tol, iter_max=self.iter_max,
                    normalisation=self.normalisation,
                    verbosity=self.verbosity, **kwargs)

        elif self.method == 'natural gradient':
            def estimation_function(X, init, **kwargs):
                return complex_tyler_shape_matrix_naturalgradient(
                    X, init=init, normalisation=self.normalisation,
                    verbosity=self.verbosity, **kwargs)
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
            X, init, **kwargs)
        self._set_covariance(covariance)
        self.err_ = err
        self.iteration_ = iteration
        return self

    def score(self, X_test, y=None, model='Gaussian'):
        # TODO: log-likelihood and all for elliptical distributions
        if model == 'Gaussian':
            return super().score(X_test)
        else:
            raise NotImplementedError(
                f"Model {model} is not known.")
