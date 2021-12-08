'''
File: robust_shape_estimation.py
File Created: Friday, 9th July 2021 10:07:35 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Wednesday, 8th December 2021 2:52:43 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import numpy as np
from robuststats.estimation.elliptical import ComplexTylerShapeMatrix, \
    get_normalisation_function, ComplexCenteredStudentMLE
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
    covariance = ToeplitzMatrix(0.9+0.4j, n_features, dtype=complex)
    # covariance = covariance / S(covariance)

    model = complex_multivariate_t(shape=covariance, df=df)
    X = model.rvs(size=n_samples)

    print("Performing Tyler fixed-point estimation of covariance matrix")
    estimator = ComplexTylerShapeMatrix(normalisation=normalisation, verbosity=True)
    estimator.fit(X)
    
    print("Performing Student-t MLE estimation of covariance matrix")
    estimator_student = ComplexCenteredStudentMLE(df=3, verbosity=True, iter_max=1000)
    estimator_student.fit(X)

    print("Saving plots to Robust_shape_estimation.png")
    fig, axes = plt.subplots(1, 3, figsize=(21, 9))
    im = axes[0].imshow(np.abs(covariance), aspect='auto')
    axes[0].set_title("True Covariance")
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(np.abs(estimator.covariance_), aspect='auto')
    axes[1].set_title(f"Estimated Covariance with Tyler $N={n_samples}$")
    fig.colorbar(im, ax=axes[1])
    
    im = axes[2].imshow(np.abs(estimator_student.covariance_), aspect='auto')
    axes[2].set_title(f"Estimated Covariance with Student-t MLE $N={n_samples}$")
    fig.colorbar(im, ax=axes[2])
    
    # plt.show()
    plt.savefig("./results/Robust_shape_estimation.png")
