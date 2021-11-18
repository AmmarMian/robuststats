'''
File: complex_autograd_tyler.py
File Created: Tuesday, 2nd November 2021 10:44:53 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 18th November 2021 4:28:02 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd
from pymanopt.function import Callable
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, ConjugateGradient
from pymanopt.manifolds.hpd import SpecialHermitianPositiveDefinite, HermitianPositiveDefinite
from robuststats.estimation.elliptical import get_normalisation_function, ComplexTylerShapeMatrix
from robuststats.models.probability import complex_multivariate_normal
from robuststats.utils.linalg import ToeplitzMatrix, invsqrtm

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
        q = np.einsum('ij,ji->i', np.conjugate(X)@la.inv(Q), X.T)
        res = p*np.sum(np.log(np.real(q))) + n*np.log(np.real(la.det(Q)))
        return res

    return cost


def _generate_egrad(X, cost):
    """Generate euclidean gradient corresponding to Tyler cost function.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dataset.
    cost :
        cost_function depending on the data X.

    Returns
    -------
    callable
        function to compute the cost at given data by X
    """

    n, p = X.shape

    @Callable
    def egrad(Q):
        res = autograd.grad(cost, argnum=[0])(Q)
        return np.conjugate(res[0])

    return egrad


if __name__ == "__main__":
    n_features = 100
    n_samples = 1000
    S = get_normalisation_function("determinant")
    covariance = ToeplitzMatrix(0.4+1j*0.4, n_features)
    covariance = covariance / S(covariance)

    print("Generating data")
    X = complex_multivariate_normal.rvs(cov=covariance, size=n_samples)
    # Normalisation to go to CAE model
    norm_X = la.norm(X, axis=1)
    X = X / np.tile(norm_X.reshape(n_samples, 1), [1, n_features])

    # Estimating using Tyler's shape matrix estimator
    print("Estimating using Tyler's shape matrix estimator")
    estimator = ComplexTylerShapeMatrix(normalisation="determinant", verbosity=True)
    estimator.fit(X)
    Q_fp = estimator.covariance_

    # Pymanopt setting
    print("Setting up pymanopt")
    manifold = SpecialHermitianPositiveDefinite(n_features)
    cost = _generate_cost_function(X)
    egrad = _generate_egrad(X, cost)
    problem = Problem(manifold=manifold, cost=cost, egrad=egrad)
    solver = SteepestDescent()

    # Solving problem
    print("Doing optimisation pymanopt")
    Qopt = solver.solve(problem)

    print("Saving plots to Tyler_gradient_estimation.png")
    fig, axes = plt.subplots(1, 3, figsize=(26, 9))
    im = axes[0].imshow(np.abs(covariance), aspect='auto')
    axes[0].set_title("True Covariance")
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(np.abs(Q_fp), aspect='auto')
    axes[1].set_title(f"Estimated Covariance with Fixed point $N={n_samples}$")
    fig.colorbar(im, ax=axes[1])

    im = axes[2].imshow(np.abs(Qopt), aspect='auto')
    axes[2].set_title(f"Estimated Covariance with gradient descent $N={n_samples}$")
    fig.colorbar(im, ax=axes[2])
    # plt.show()
    plt.savefig("Complex_Tyler_gradient_estimation.png")
