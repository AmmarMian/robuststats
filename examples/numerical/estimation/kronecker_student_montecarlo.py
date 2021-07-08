'''
File: kronecker_student_montecarlo.py
File Created: Thursday, 8th July 2021 3:31:21 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 8th July 2021 3:57:53 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''


import numpy as np
import pandas as pd
from tqdm import trange
import plotly.express as px
from joblib import Parallel, delayed
from robuststats.estimation.base import ComplexEmpiricalCovariance
from robuststats.estimation.structured import KroneckerStudent
from robuststats.models.probability import complex_multivariate_t
from robuststats.utils.montecarlo import temp_seed
# from robuststats.utils.verbose import matprint
from pyCovariance.generation_data import generate_complex_covariance


def monte_carlo_trial(trial, a, b, mean, covariance, d,
                      n_samples_list, base_seed=77):
    # Estimators
    scm = ComplexEmpiricalCovariance()
    kronecker = KroneckerStudent(a, b, d)
    estimator_list = [scm, kronecker]

    with temp_seed(base_seed + trial):
        number_of_points = len(n_samples_list)
        error_array = np.full((number_of_points, len(estimator_list)), np.nan)
        for i, n_samples in enumerate(n_samples_list):
            X = complex_multivariate_t.rvs(mean, covariance, d, size=n_samples)
            for j, estimator in enumerate(estimator_list):
                estimator.fit(X)
                error_array[i, j] = np.linalg.norm(
                    covariance - estimator.covariance_, 'fro') / \
                    np.linalg.norm(covariance, 'fro')
        return error_array


if __name__ == '__main__':

    # Deterministic execution
    base_seed = 77
    np.random.seed(base_seed)

    # Monte_carlo parameters
    n_trials = 1000
    number_of_points = 15

    # Data parameters
    a = 3
    b = 9
    n_features = a*b
    A = generate_complex_covariance(a, unit_det=True)
    B = generate_complex_covariance(b)
    mean = np.zeros((n_features,))
    covariance = np.kron(A, B)
    mean = np.zeros((n_features,), dtype=complex)
    n_samples_list = np.unique(np.logspace(1.1, 3, number_of_points,
                                           base=n_features, dtype=int))

    # Performing Monte-carlo in Student-t
    print("Performing simulation")
    df = 3
    error = Parallel(n_jobs=-1)(
        delayed(monte_carlo_trial)(
            trial, a, b, mean, covariance, df, n_samples_list)
        for trial in trange(n_trials)
        )
    error = np.mean(np.dstack(error), axis=2)
    df = pd.DataFrame(
        data=np.hstack([n_samples_list.reshape(number_of_points, 1), error]),
        columns=["N", "SCM", "Kronecker gradient"])

    print("Saving plot")
    fig = px.scatter(
        df, x="N", y=["SCM", "Kronecker gradient"], log_x=True, log_y=True
    )
    fig.write_html("error_kronecker_student.html")
