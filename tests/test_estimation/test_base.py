'''
File: test_base.py
File Created: Sunday, 20th June 2021 7:11:16 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Wednesday, 8th December 2021 5:18:31 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

from robuststats.models.mappings import check_Hermitian
from robuststats.estimation.base import ComplexEmpiricalCovariance,\
    complex_empirical_covariance, CovariancesEstimation
from robuststats.estimation.elliptical import TylerShapeMatrix, \
    ComplexTylerShapeMatrix

import numpy.random as rnd
import numpy.testing as np_test
import numpy as np

import logging
logging.basicConfig(level='INFO')

# ----------------------------------------
# Test Transformer classes
# ----------------------------------------
def test_CovariancesEstimation_real():
    seed=777
    n_trials, n_samples, n_features = 100, 1000, 27
    X = np.random.randn(n_trials, n_samples, n_features)
    estimator = TylerShapeMatrix()
    
    # Standard without parallel
    transformer = CovariancesEstimation(estimator)
    covmats = transformer.transform(X)
    
    assert np.isrealobj(covmats)
    assert covmats.shape == (n_trials, n_features, n_features)
    
    # Standard with parallel
    transformer = CovariancesEstimation(estimator, n_jobs=-1)
    covmats = transformer.transform(X)
    
    assert np.isrealobj(covmats)
    assert covmats.shape == (n_trials, n_features, n_features)


def test_CovariancesEstimation_complex():
    seed=777
    n_trials, n_samples, n_features = 100, 1000, 27
    X = np.random.randn(n_trials, n_samples, n_features) +1j*np.random.randn(n_trials, n_samples, n_features)
    estimator = ComplexTylerShapeMatrix()
    
    # Standard without parallel
    transformer = CovariancesEstimation(estimator)
    covmats = transformer.transform(X)
    
    assert np.iscomplexobj(covmats)
    assert covmats.shape == (n_trials, n_features, n_features)
    
    # With parallel
    transformer = CovariancesEstimation(estimator, n_jobs=-1)
    covmats = transformer.transform(X)
    
    assert np.iscomplexobj(covmats)
    assert covmats.shape == (n_trials, n_features, n_features)
    
# ----------------------------------------
# Test Estimation classes
# ----------------------------------------
def test_ComplexEmpiricalCovariance():
    seed = 761
    rnd.seed(seed)

    n_features = 17
    n_samples = 200

    a = np.random.randn(n_samples, n_features) + \
        1j*np.random.randn(n_samples, n_features)
    estimator = ComplexEmpiricalCovariance()
    estimator.fit(a)
    covariance = estimator.covariance_
    covariance_from_function = complex_empirical_covariance(a)

    assert np.iscomplexobj(covariance)
    assert covariance.shape == (n_features, n_features)
    assert check_Hermitian(covariance)
    assert check_Hermitian(covariance_from_function)
    np_test.assert_array_equal(covariance_from_function, covariance)
    assert estimator.error_norm(covariance) == 0
    assert estimator.error_norm(covariance + np.eye(n_features)) >= 0
    estimator._set_covariance(covariance + np.eye(n_features))
    np_test.assert_array_equal(covariance + np.eye(n_features),
                               estimator.covariance_)
