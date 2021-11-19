'''
File: natural_gradient_tyler.py
File Created: Tuesday, 2nd November 2021 10:44:53 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Friday, 19th November 2021 5:23:44 pm
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
from pymanopt.manifolds.psd import SymmetricPositiveDefinite
from robuststats.estimation.elliptical import get_normalisation_function, TylerShapeMatrix
from scipy.stats import multivariate_normal
from robuststats.utils.linalg import ToeplitzMatrix, invsqrtm

import matplotlib.pyplot as plt



if __name__ == "__main__":
    n_features = 100
    n_samples = 1000
    S = get_normalisation_function("determinant")
    covariance = ToeplitzMatrix(0.85, n_features, dtype=float)
    covariance = covariance / S(covariance)

    print("Generating data")
    X = multivariate_normal.rvs(cov=covariance, size=n_samples)
    # Normalisation to go to CAE model
    norm_X = la.norm(X, axis=1)
    X = X / np.tile(norm_X.reshape(n_samples, 1), [1, n_features])

    # Estimating using fixed-point Tyler's shape matrix estimator
    print("Estimating using fixed point Tyler's shape matrix estimator")
    estimator = TylerShapeMatrix(normalisation="determinant", verbosity=True)
    estimator.fit(X)
    Q_fp = estimator.covariance_


    # Estimating using natural gradient Tyler's shape matrix estimator
    print("Estimating using natural gradient Tyler's shape matrix estimator")
    estimator = TylerShapeMatrix(method="natural gradient",
        normalisation="determinant", verbosity=True)
    estimator.fit(X)
    Qopt = estimator.covariance_

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
    plt.savefig("./results/Tyler_gradient_estimation.png")
