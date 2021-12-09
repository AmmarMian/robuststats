'''
File: complex_natural_gradient_tyler.py
File Created: Wednesday, 8th December 2021 10:50:44 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 9th December 2021 3:53:11 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''


import numpy as np
from robuststats.estimation.covariance import get_normalisation_function, ComplexTylerShapeMatrix
from robuststats.models.probability import complex_multivariate_t
from robuststats.utils.generation_data import generate_complex_covariance
from robuststats.utils.linalg import invsqrtm, ToeplitzMatrix

import matplotlib.pyplot as plt



if __name__ == "__main__":
    n_features = 250
    n_samples = 5000
    S = get_normalisation_function("determinant")
    covariance = ToeplitzMatrix(0.8+0.4j, n_features, dtype=np.complex128)
    covariance = covariance / S(covariance)

    print("Generating data")
    X = complex_multivariate_t.rvs(shape=covariance, df=30, size=n_samples)


    # Estimating using fixed-point Tyler's shape matrix estimator
    print("Estimating using fixed point Tyler's shape matrix estimator")
    estimator = ComplexTylerShapeMatrix(normalisation="determinant", verbosity=True)
    estimator.fit(X)
    Q_fp = estimator.covariance_


    # Estimating using natural gradient Tyler's shape matrix estimator
    print("Estimating using natural gradient Tyler's shape matrix estimator")
    estimator = ComplexTylerShapeMatrix(method="natural gradient",
        normalisation="determinant", verbosity=True)
    estimator.fit(X)
    Qopt = estimator.covariance_

    print("Saving plots to Complex_Tyler_gradient_estimation.png")
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
    plt.savefig("./results/Complex_Tyler_gradient_estimation.png")
