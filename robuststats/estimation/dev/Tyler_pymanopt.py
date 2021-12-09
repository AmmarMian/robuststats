'''
File: Tyler_pymanopt.py
File Created: Friday, 9th July 2021 11:04:56 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 9th December 2021 3:53:11 pm
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
from robuststats.estimation.covariance import get_normalisation_function, TylerShapeMatrix
from pymanopt.manifolds.psd import SymmetricPositiveDefinite
from pyCovariance.matrix_operators import invsqrtm

from scipy.stats import multivariate_normal
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

    n, p = X.shape

    @Callable
    def cost(Q):
        temp = invsqrtm(Q)@X.T
        q = np.einsum('ij,ji->i', temp.T, temp)
        return p*np.sum(np.log(q)) + n*np.log(la.det(Q))

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

    n, p = X.shape

    @Callable
    def egrad(Q):
        i_Q = la.inv(Q)
        i2_Q = la.matrix_power(i_Q, 2)

        result = 0
        for i in range(n):
            # result -= p * (X[i, :].T@X[i, :])/np.trace(X[i, :].T@X[i, :]@i_Q) @ i2_Q
            x_i = X[i, :].T.reshape((p, 1))
            result -= p * (x_i@x_i.T)/np.trace((x_i@x_i.T)@i_Q)
        return result @ i2_Q + n*i_Q

        # temp = invsqrtm(Q)@X.T
        # tau = np.einsum('ij,ji->i', temp.T, temp)
        # tau = (1/p) * tau
        # temp = X.T / np.sqrt(tau)

        # return -(temp@temp.T)@i2_Q + n*i_Q

    return egrad


if __name__ == '__main__':

    # TODO: Debug
    # Data generation
    n_features = 100
    n_samples = 10000
    S = get_normalisation_function("determinant")
    covariance = ToeplitzMatrix(0.95, n_features, dtype=float)
    covariance = covariance / S(covariance)

    print("Generating data")
    X = multivariate_normal.rvs(cov=covariance, size=n_samples)

    # Estimating using Tyler's shape matrix estimator
    print("Estimating using Tyler's shape matrix estimator")
    estimator = TylerShapeMatrix(normalisation="determinant", verbosity=True)
    estimator.fit(X)
    Q_fp = estimator.covariance_

    # Normalisation to go to CAE model
    norm_X = la.norm(X, axis=1)
    X = X / np.tile(norm_X.reshape(n_samples, 1), [1, n_features])

    # Pymanopt setting
    print("Setting up pymanopt")
    manifold = SymmetricPositiveDefinite(n_features)
    cost = _generate_cost_function(X)
    egrad = _generate_egrad(X)
    problem = Problem(manifold=manifold, cost=cost, egrad=egrad)
    solver = SteepestDescent()

    # Solving problem
    print("Doing optimisation pymanopt")
    Qopt = solver.solve(problem)

    print("Saving plots to Tyler_gradient_estimation.png")
    fig, axes = plt.subplots(1, 3, figsize=(26, 9))
    im = axes[0].imshow(covariance, aspect='auto')
    axes[0].set_title("True Covariance")
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(Q_fp, aspect='auto')
    axes[1].set_title(f"Estimated Covariance with Fixed point $N={n_samples}$")
    fig.colorbar(im, ax=axes[1])

    im = axes[2].imshow(Qopt, aspect='auto')
    axes[2].set_title(f"Estimated Covariance with gradient descent $N={n_samples}$")
    fig.colorbar(im, ax=axes[2])
    # plt.show()
    plt.savefig("Tyler_gradient_estimation.png")

    # # Handmade gradient descent euclidean
    # print("Doing euclidean gradient descent")
    # alpha = 0.05
    # estimate = np.eye(n_features)
    # converged = False
    # delta = 1e-4
    # while not converged:
    #     new_estimate = estimate - alpha*egrad(estimate)
    #     err = np.linalg.norm(estimate - new_estimate)/np.linalg.norm(estimate)
    #     estimate = new_estimate
    #     converged = err < delta
    #     print(err)
