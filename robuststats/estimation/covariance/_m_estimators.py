'''
File: _m_estimators.py
File Created: Tuesday, 21st December 2021 10:02:08 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 21st December 2021 5:27:37 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''


import numpy as np
import numpy.linalg as la
import logging
from tqdm import tqdm

from sklearn.covariance import empirical_covariance
from .base import complex_empirical_covariance, get_normalisation_function
from ...utils.linalg import invsqrtm

    

# -----------------------------------------------------------------------
# M-estimators functions
# -----------------------------------------------------------------------
def _student_t_m_estimator_function(x, df=3, n_features=1):
    """Student-t mle m-estimator function

    Parameters
    ----------
    x : array-like
        input data
    df : int, optional
        degrees of freedom of Student-t law, by default 3
    n_features : int, optional
        optional for compatibility with kwargs but nut so much
        optional, by default 1
    """
    return (n_features+df/2)/(x+df/2)


def _huber_m_estimator_function(x, lbda=np.inf, beta=1, **kwargs):
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
    lbda : float, optional
        value at which we shift from no ponderation to inverse ponderation. By default inf.
    beta : float, optional
        a tuning parameter. By default, 1.

    Returns
    -------
    array-like
        array of the same shape as input with values that have been scaled depending
        on the tuning parameters

    """
    if lbda<=0 or beta<=0:
        raise AssertionError(
            f"Error, lambda or beta can't be negative or null : lambda={lbda}, beta={beta}"
        )
    return (1/beta)*np.minimum(1,lbda/x)


def _tyler_m_estimator_function(x, n_features=1, **kwargs):
    """Tyler M-estimator function

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        quadratic form
    n_features : int, optional
        optional for compatibility with kwargs but nut so much
        optional, by default 1

    Returns
    -------
    array-like of shape (n_samples,)
        scaled quadratic form to have Tyler's estimator
    """
    return n_features/x


# -----------------------------------------------------------------------
# Algorithms
# -----------------------------------------------------------------------
def fixed_point_m_estimation_centered(X, m_estimator_function, init=None,
                                      tol=1e-4, iter_max=30, 
                                      verbosity=False,  **kwargs):
    """Fixed-point algorithm for an arbitrary M-estimators of
    covariance matrix as defined in:
    >Ricardo Antonio Maronna.
    >"Robust $M$-Estimators of Multivariate Location and Scatter." The Annals of Statistics, 4(1) 51-67 January, 1976.
    >https://doi.org/10.1214/aos/1176343347

    Data is assumed to be centered.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
          n_features is the number of features.
    m_estimator_function : function
        Scaling to apply to the quadratic form in the M-estimating
        fixed-point equation. function has to be vectorized.
    init : array-like of shape (n_features, n_features), optional
        Initial point of algorithm, by default None.
    tol : float, optional
        tolerance for convergence of algorithm, by default 1e-4.
    iter_max : int, optional
        number of maximum iterations, by default 30.
    verbosity : bool, optional
        show progress of algorithm at each iteration, by default False
    **kwargs : 
        arguments to m_estimator_function
    """
    
    # Checking whether data is real or complex-valued
    if np.iscomplexobj(X):
        tp = lambda x: np.conjugate(np.transpose(x))
        init_function = complex_empirical_covariance
    else:
        tp = lambda x: np.transpose(x)
        init_function = empirical_covariance
        
    # Initialisation
    N, p = X.shape

    if init is None:
        sigma = init_function(X, assume_centered=True)
    else:
        sigma = init.astype(X.dtype)
        
    # Fixed-point loop
    delta = np.inf  # Distance between two iterations
    iteration = 0

    if verbosity:
        pbar_v = tqdm(total=iter_max)

    while (delta > tol) and (iteration < iter_max):
        # compute expression of Tyler estimator
        temp = invsqrtm(sigma)@X.T
        quadratic = np.einsum('ij,ji->i', tp(temp), temp)
        quadratic = m_estimator_function(np.real(quadratic), **kwargs)
        temp = X.T * np.sqrt(quadratic)
        sigma_new = (1/N) * temp@tp(temp)

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
    