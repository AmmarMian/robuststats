'''
File: robust_shape_estimation.py
File Created: Friday, 9th July 2021 10:07:35 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Friday, 9th July 2021 11:03:29 am
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import numpy as np
from robuststats.estimation.elliptical import TylerShapeMatrix, get_normalisation_function
from robuststats.models.probability import complex_multivariate_t
from robuststats.utils.linalg import ToeplitzMatrix

import matplotlib.pyplot as plt
import logging
logging.basicConfig(level='INFO')

if __name__ == '__main__':

    # Deterministic execution
    base_seed = 177
    np.random.seed(base_seed)

    # Data parameters
    n_features = 100
    n_samples = 10000

    print("Generating data")
    df = 3
    normalisation = "determinant"
    S = get_normalisation_function(normalisation)
    covariance = ToeplitzMatrix(0.9, n_features, dtype=complex)
    covariance = covariance / S(covariance)

    model = complex_multivariate_t(shape=covariance, df=df)
    X = model.rvs(size=n_samples)

    print("Performing Tyler fixed-point estimation of covariance matrix")
    estimator = TylerShapeMatrix(normalisation=normalisation, verbosity=True)
    estimator.fit(X)

    print("Saving plots to Tyler_estimation.png")
    fig, axes = plt.subplots(1, 2, figsize=(21, 9))
    im = axes[0].imshow(np.abs(covariance), aspect='auto')
    axes[0].set_title("True Covariance")
    fig.colorbar(im, ax=axes[0])

    axes[1].imshow(np.abs(estimator.covariance_), aspect='auto')
    axes[1].set_title(f"Estimated Covariance $N={n_samples}$")
    fig.colorbar(im, ax=axes[1])
    # plt.show()
    plt.savefig("Tyler_estimation.png")
