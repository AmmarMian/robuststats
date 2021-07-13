'''
File: kronecker_real.py
Created Date: Tuesday July 13th 2021 - 03:05pm
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Tue Jul 13 2021
Modified By: Ammar Mian
-----
Copyright (c) 2021 Université Savoie Mont-Blanc
'''


import numpy as np
import numpy.linalg as la
from pymanopt.function import Callable
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, ConjugateGradient
from pymanopt.manifolds.product import Product
from pymanopt.manifolds.psd import SymmetricPositiveDefinite
from robuststats.utils.linalg import hermitian, ToeplitzMatrix
from scipy.stats import multivariate_t
from scipy.linalg import orth

import matplotlib.pyplot as plt

if __name__ == '__main__':

    results_directory = "/Users/ammarmian/Desktop/temp/"

    # ----------------------------------------------------------------
    # Data Generation
    # ----------------------------------------------------------------
    np.random.seed(75)
    a = 4
    b = 4
    n_features = a*b
    df = 3
    alpha = (df+n_features)/(df+n_features+1)
    manifold_A = SymmetricPositiveDefinite(a)
    manifold_B = SymmetricPositiveDefinite(b)
    manifold = Product((manifold_A, manifold_B))
    
    # Setting matrices 
    cA = 10 
    UA = orth(np.random.randn(a, a))
    sA = np.zeros((a))
    sA[0] = 1/np.sqrt(cA)
    sA[1] = np.sqrt(cA)
    if a>2:
        sA[2:] = 1/np.sqrt(cA) + (np.sqrt(cA) - 1/np.sqrt(cA))*np.random.rand(a-2)
    A = UA@np.diag(sA)@UA.T
    A = A / np.linalg.det(A)**(1/a)

    cB = 10
    UB = orth(np.random.randn(b, b))
    sB = np.zeros((b))
    sB[0] = 1/np.sqrt(cB)
    sB[1] = np.sqrt(cB)
    if a>2:
        sB[2:] = 1/np.sqrt(cB) + (np.sqrt(cB) - 1/np.sqrt(cB))*np.random.rand(b-2)
    B = UB@np.diag(sB)@UB.T

    covariance = np.kron(A, B)

    n_samples = 10000
    model = multivariate_t(shape=covariance, df=df)
    X = model.rvs(size=n_samples)

    # ----------------------------------------------------------------
    # Gradient estimation
    # ----------------------------------------------------------------
    @Callable
    def cost_function(A, B):
        return - np.sum(multivariate_t.logpdf(X, shape=np.kron(A,B), df=df))

    @Callable
    def egrad_function(A, B):
        # Pre-calculations
        i_A = la.inv(A)
        i_B = la.inv(B)

        grad_A = np.zeros_like(A)
        grad_B = np.zeros_like(B)
        for i in range(n_samples):
            M_i = X[i, :].reshape((b, a), order='F')
            Q_i = np.real(np.trace(i_A@M_i.T@i_B@M_i))

            grad_A -= (df+a*b)/(df+Q_i) * i_A@(M_i.T@i_B@M_i)@i_A
            grad_B -= (df+a*b)/(df+Q_i) * i_B@(M_i@i_A@M_i.T)@i_B
        grad_A = grad_A/n_samples + b*i_A
        grad_B = grad_B/n_samples + a*i_B

        return (grad_A, grad_B)

    print("Performing gradient estimation")
    problem = Problem(
        manifold=manifold, cost=cost_function, egrad=egrad_function, verbosity=2
    )
    solver = ConjugateGradient()
    theta_0 = (np.eye(a), np.eye(b))
    A_opt, B_opt = solver.solve(problem, x=theta_0)

    fig, axes = plt.subplots(1, 2, figsize=(21, 9))
    im = axes[0].imshow(np.abs(covariance), aspect='auto')
    axes[0].set_title("True Covariance")
    fig.colorbar(im, ax=axes[0])
    axes[1].imshow(np.abs(np.kron(A_opt, B_opt)), aspect='auto')
    axes[1].set_title(f"Estimated Covariance $N={n_samples}$")
    fig.colorbar(im, ax=axes[1])
    plt.savefig(f"{results_directory}/gradient.png")

 