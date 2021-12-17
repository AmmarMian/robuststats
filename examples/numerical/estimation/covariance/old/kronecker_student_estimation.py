'''
File: kronecker_student_estimation.py
File Created: Thursday, 8th July 2021 3:18:13 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tue Jul 13 2021
Modified By: Ammar Mian
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import numpy as np
import numpy.linalg as la
from robuststats.estimation.structured import KroneckerStudent,\
    _generate_egrad_function_kronecker_student, _generate_cost_function_kronecker_student, KroneckerEllipticalMM
from robuststats.models.manifolds import KroneckerHermitianPositiveElliptical
from robuststats.models.probability import complex_multivariate_t
from robuststats.online.estimation import StochasticGradientCovarianceEstimator
from robuststats.utils.linalg import hermitian, ToeplitzMatrix

import matplotlib.pyplot as plt

if __name__ == '__main__':

    results_directory = "/Users/ammarmian/Desktop/temp/"

    # ----------------------------------------------------------------
    # Data Generation
    # ----------------------------------------------------------------
    np.random.seed(75)
    a = 7
    b = 9
    n_features = a*b
    df = 3
    alpha = (df+n_features)/(df+n_features+1)
    manifold = KroneckerHermitianPositiveElliptical(a, b, alpha)
    A = ToeplitzMatrix(0.3, a)
    B = ToeplitzMatrix(0.1, b)
    covariance = np.kron(A, B)

    n_samples = 10000
    model = complex_multivariate_t(shape=covariance, df=df)
    X = model.rvs(size=n_samples)

    # ----------------------------------------------------------------
    # Gradient estimation
    # ----------------------------------------------------------------
    print("Performing gradient estimation")
    kronecker_gradient = KroneckerStudent(a, b, df, verbosity=2)
    kronecker_gradient.fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(21, 9))
    im = axes[0].imshow(np.abs(covariance), aspect='auto')
    axes[0].set_title("True Covariance")
    fig.colorbar(im, ax=axes[0])
    axes[1].imshow(np.abs(kronecker_gradient.covariance_), aspect='auto')
    axes[1].set_title(f"Estimated Covariance $N={n_samples}$")
    fig.colorbar(im, ax=axes[1])
    plt.savefig(f"{results_directory}/gradient.png")

    # ----------------------------------------------------------------
    # MM estimation
    # ----------------------------------------------------------------
    print("Performing MM estimation")
    kronecker_mm = KroneckerEllipticalMM(a, b, verbosity=True)
    kronecker_mm.fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(21, 9))
    im = axes[0].imshow(np.abs(covariance), aspect='auto')
    axes[0].set_title("True Covariance")
    fig.colorbar(im, ax=axes[0])
    axes[1].imshow(np.abs(kronecker_mm.covariance_), aspect='auto')
    axes[1].set_title(f"Estimated Covariance $N={n_samples}$")
    fig.colorbar(im, ax=axes[1])
    plt.savefig(f"{results_directory}/MM.png")

    # ----------------------------------------------------------------
    # Online estimation
    # ----------------------------------------------------------------
    print("Performing online estimation")

    def estimatetocov(estimate):
        A, B = estimate
        return np.kron(A, B)

    egrad = _generate_egrad_function_kronecker_student(X, a, b, df)

    def proj(A, xi):
        return xi - np.trace(la.inv(A)@xi)*A/a

    def rgrad(X, estimate):
        A, B = estimate
        e_A, e_B = egrad(A, B)

        xi = A@hermitian(e_A)@A
        r_A = proj(A, xi) / (alpha*b)
        r_B = B@hermitian(e_B)@B / (alpha*a) -\
            (alpha - 1)*np.trace(B@e_B)*B /\
            (alpha*(alpha+(alpha - 1)*n_features))

        return r_A, r_B

    def cost(X, estimate):
        A, B = estimate
        _cost = _generate_cost_function_kronecker_student(X, df)
        return _cost(A, B)

    # # TODO: debug why become not spd at some point !!!
    kronecker_online = StochasticGradientCovarianceEstimator(
        manifold, rgrad, 0.95, estimatetocov, cost, verbosity=True
    )
    init = (np.eye(a, dtype=complex), np.eye(b, dtype=complex))
    kronecker_online.fit(X, init=init)

    fig, axes = plt.subplots(1, 2, figsize=(21, 9))
    im = axes[0].imshow(np.abs(covariance), aspect='auto')
    axes[0].set_title("True Covariance")
    fig.colorbar(im, ax=axes[0])
    axes[1].imshow(np.abs(kronecker_online.covariance_), aspect='auto')
    axes[1].set_title(f"Estimated Covariance $N={n_samples}$")
    fig.colorbar(im, ax=axes[1])
    plt.savefig(f"{results_directory}/online.png")
