'''
File: robust_shape_montecarlo.py
File Created: Wednesday, 7th July 2021 1:07:17 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 9th December 2021 3:53:11 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Universit√© Savoie Mont-Blanc
'''

import numpy as np
import pandas as pd
from tqdm import trange
import plotly.express as px
from joblib import Parallel, delayed
from robuststats.estimation.base import ComplexEmpiricalCovariance
from robuststats.estimation.covariance import ComplexTylerShapeMatrix
from robuststats.models.probability import complex_multivariate_normal,\
                                           complex_multivariate_t
from robuststats.utils.montecarlo import temp_seed
# from robuststats.utils.verbose import matprint
from robuststats.utils.generation_data import generate_complex_covariance


def monte_carlo_trial_gaussian(trial, mean, covariance,
                               n_samples_list, base_seed=77):

    # Estimators
    scm = ComplexEmpiricalCovariance()
    tyler = ComplexTylerShapeMatrix(normalisation='determinant',
                                    verbosity=False)
    estimator_list = [scm, tyler]

    with temp_seed(base_seed + trial):
        number_of_points = len(n_samples_list)
        error_array = np.full((number_of_points, len(estimator_list)), np.nan)
        for i, n_samples in enumerate(n_samples_list):
            X = complex_multivariate_normal.rvs(
                mean=mean, cov=covariance, size=n_samples)
            for j, estimator in enumerate(estimator_list):
                estimator.fit(X)
                error_array[i, j] = np.linalg.norm(
                    covariance - estimator.covariance_, 'fro') / \
                    np.linalg.norm(covariance, 'fro')
        return error_array


def monte_carlo_trial_student(trial, mean, covariance, d,
                              n_samples_list, base_seed=77):
    # Estimators
    scm = ComplexEmpiricalCovariance()
    tyler = ComplexTylerShapeMatrix(normalisation='determinant',
                                    verbosity=False)
    estimator_list = [scm, tyler]

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
    n_features = 17
    mean = np.zeros((n_features,), dtype=complex)
    covariance = generate_complex_covariance(n_features, unit_det=True)
    n_samples_list = np.unique(np.logspace(1.1, 3, number_of_points,
                                           base=n_features, dtype=int))

    # Performing Monte-carlo in Gaussian
    print("Performing Gaussian simulation")
    error = Parallel(n_jobs=-1)(
        delayed(monte_carlo_trial_gaussian)(
            trial, mean, covariance, n_samples_list)
        for trial in trange(n_trials))
    error = np.mean(np.dstack(error), axis=2)
    df = pd.DataFrame(
        data=np.hstack([n_samples_list.reshape(number_of_points, 1), error]),
        columns=["N", "SCM", "Tyler"])

    fig = px.scatter(df, x="N", y=["SCM", "Tyler"], log_x=True, log_y=True)
    fig.write_html("./results/error_Gaussian.html")

    # Performing Monte-carlo in Student-t pseudo-Gaussian
    print("Performing pesudo-Gaussian simulation")
    d = 50
    error = Parallel(n_jobs=-1)(
        delayed(monte_carlo_trial_student)(
            trial, mean, covariance, d, n_samples_list)
        for trial in trange(n_trials))
    error = np.mean(np.dstack(error), axis=2)
    df = pd.DataFrame(
        data=np.hstack([n_samples_list.reshape(number_of_points, 1), error]),
        columns=["N", "SCM", "Tyler"])

    fig = px.scatter(df, x="N", y=["SCM", "Tyler"], log_x=True, log_y=True)
    fig.write_html("./results/error_pesudo_Gaussian.html")

    # Performing Monte-carlo in Student-t heavy-tailed
    print("Performing heavy-tailed simulation")
    d = 2
    error = Parallel(n_jobs=-1)(
        delayed(monte_carlo_trial_student)(
            trial, mean, covariance, d, n_samples_list)
        for trial in trange(n_trials))
    error = np.mean(np.dstack(error), axis=2)
    df = pd.DataFrame(
        data=np.hstack([n_samples_list.reshape(number_of_points, 1), error]),
        columns=["N", "SCM", "Tyler"])

    fig = px.scatter(df, x="N", y=["SCM", "Tyler"], log_x=True, log_y=True)
    fig.write_html("./results/error_heavy_tailed.html")
