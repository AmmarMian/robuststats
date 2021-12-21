'''
File: test_base.py
File Created: Sunday, 20th June 2021 7:11:16 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 21st December 2021 5:11:42 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

from robuststats.models.mappings import check_Hermitian
from robuststats.estimation.covariance.base import ComplexEmpiricalCovariance,\
    complex_empirical_covariance, CovariancesEstimation, get_normalisation_function
from robuststats.estimation.covariance import TylerShapeMatrix, ComplexTylerShapeMatrix
from robuststats.utils.generation_data import generate_complex_covariance, generate_covariance
from robuststats.models.probability import complex_multivariate_normal
import numpy.random as rnd
import numpy.testing as np_test
import numpy as np
import numpy.linalg as la

import logging
logging.basicConfig(level='INFO')

# ----------------------------------------
# Test functions
# ----------------------------------------
def test_get_normalisation_function_real_values():
    seed = 761
    rnd.seed(seed)
    trace_function = get_normalisation_function('trace')
    determinant_function = get_normalisation_function('determinant')
    element_function = get_normalisation_function('element')
    none_function = get_normalisation_function()

    n_features = 17
    covariance = generate_covariance(n_features)

    assert trace_function(covariance) == np.trace(covariance) / n_features
    assert determinant_function(covariance) ==\
        la.det(covariance)**(1/n_features)
    assert element_function(covariance) == covariance[0, 0]
    assert none_function(covariance) == 1


def test_get_normalisation_function_complex_values():
    seed = 761
    rnd.seed(seed)
    trace_function = get_normalisation_function('trace')
    determinant_function = get_normalisation_function('determinant')
    element_function = get_normalisation_function('element')
    none_function = get_normalisation_function()

    n_features = 17
    covariance = generate_complex_covariance(n_features)

    assert trace_function(covariance) == np.real(np.trace(
                                            covariance)) / n_features
    assert determinant_function(covariance) ==\
        np.real(la.det(covariance))**(1/n_features)
    assert element_function(covariance) == covariance[0, 0]
    assert none_function(covariance) == 1



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
