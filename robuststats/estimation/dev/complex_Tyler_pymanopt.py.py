'''
File: Tyler_pymanopt.py
File Created: Friday, 9th July 2021 11:04:56 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 2nd November 2021 10:37:59 am
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

# import autograd.numpy as np
# import autograd.numpy.linalg as la
import numpy as np
import numpy.linalg as la

from pymanopt.function import Autograd, Callable
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, ConjugateGradient
from robuststats.estimation.elliptical import get_normalisation_function
from pymanopt.manifolds.hpd import SpecialHermitianPositiveDefinite, HermitianPositiveDefinite
from pyCovariance.matrix_operators import invsqrtm

from robuststats.models.probability import complex_multivariate_normal
from robuststats.utils.linalg import ToeplitzMatrix

import matplotlib.pyplot as plt


def _generate_cost_function(X):
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

    p, n = X.shape

    @Callable
    def cost(Q):
        temp = invsqrtm(Q)@X.T
        temp_conj = np.real(temp) - 1j*np.imag(temp)
        q = np.real(np.einsum('ij,ji->i', temp_conj.T, temp))
        return p*np.sum(np.log(q)) + n*np.log(np.abs(la.det(Q)))

    return cost


def _generate_egrad(X):
    """Generate euclidean gradient corresponding to Tyler cost function.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.

    Returns
    -------
    callable
        function to compute the cost at given data by X
    """

    p, n = X.shape

    @Callable
    def egrad(Q):
        i_Q = la.inv(Q)
        i2_Q = la.matrix_power(i_Q, 2)

        temp = invsqrtm(Q)@X.T
        temp_conj = np.real(temp) - 1j*np.imag(temp)
        tau = np.einsum('ij,ji->i', temp.conj().T, temp)
        tau = (1/p) * np.real(tau)
        temp = X.T / np.sqrt(tau)
        temp_conj = np.real(temp) - 1j*np.imag(temp)

        return -(temp@temp_conj.T)@i2_Q  + n*i_Q

    return egrad


if __name__ == '__main__':

    #TODO: Debug

    # Data generation
    n_features = 1000
    n_samples = 10000
    S = get_normalisation_function("determinant")
    covariance = ToeplitzMatrix(0.5+1j*0.3, n_features)
    covariance = covariance / S(covariance)
    X = complex_multivariate_normal.rvs(cov=covariance, size=n_samples)

    # Normalisation to go to CAE model
    norm_X = la.norm(X, axis=1)
    X = X / np.tile(norm_X.reshape(n_samples, 1), [1, n_features])

    # Pymanopt setting
    manifold = SpecialHermitianPositiveDefinite(n_features)
    cost = _generate_cost_function(X)
    egrad = _generate_egrad(X)
    problem = Problem(manifold=manifold, cost=cost, egrad=egrad)
    solver = ConjugateGradient()

    # Solving problem
    Qopt = solver.solve(problem)

    print("Saving plots to Complex_Tyler_gradient_estimation.png")
    fig, axes = plt.subplots(1, 2, figsize=(21, 9))
    im = axes[0].imshow(np.abs(covariance), aspect='auto')
    axes[0].set_title("True Covariance")
    fig.colorbar(im, ax=axes[0])

    axes[1].imshow(np.abs(Qopt), aspect='auto')
    axes[1].set_title(f"Estimated Covariance $N={n_samples}$")
    fig.colorbar(im, ax=axes[1])
    # plt.show()
    plt.savefig("Complex_Tyler_gradient_estimation.png")
