'''
File: test_elliptical.py
File Created: Sunday, 20th June 2021 9:16:15 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 8th July 2021 3:09:01 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

from robuststats.models.mappings import check_Hermitian
from robuststats.estimation.elliptical import get_normalisation_function,\
    tyler_shape_matrix_fixedpoint, TylerShapeMatrix
from pyCovariance.generation_data import generate_complex_covariance,\
    sample_complex_normal_distribution

import numpy.random as rnd
import numpy.testing as np_test
import numpy as np
import numpy.linalg as la

import logging
logging.basicConfig(level='INFO')


def test_get_normalisation_function():
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


def test_tyler_shape_matrix_fixedpoint():
    seed = 761
    rnd.seed(seed)

    n_features = 17
    n_samples = 200

    a = np.random.randn(n_samples, n_features) + \
        1j*np.random.randn(n_samples, n_features)

    covariance, _, _ = tyler_shape_matrix_fixedpoint(a)
    assert np.iscomplexobj(covariance)
    assert covariance.shape == (n_features, n_features)
    assert check_Hermitian(covariance)

    covariance, _, _ = tyler_shape_matrix_fixedpoint(a, normalisation='trace')
    np_test.assert_equal(np.real(np.trace(covariance)), n_features)

    covariance, _, _ = tyler_shape_matrix_fixedpoint(
                                a, normalisation='determinant')
    np_test.assert_almost_equal(
                        np.real(la.det(covariance)), 1)

    covariance, _, _ = tyler_shape_matrix_fixedpoint(
                                    a, normalisation='element')
    np_test.assert_almost_equal(covariance[0, 0], 1)


def test_TylerShapeMatrix():
    seed = 761
    rnd.seed(seed)

    n_features = 17
    n_samples = 200

    a = np.random.randn(n_samples, n_features) + \
        1j*np.random.randn(n_samples, n_features)
    estimator = TylerShapeMatrix()
    estimator.fit(a)
    covariance = estimator.covariance_
    assert np.iscomplexobj(covariance)
    assert covariance.shape == (n_features, n_features)
    assert check_Hermitian(covariance)

    estimator = TylerShapeMatrix(tol=1e-4, iter_max=100, normalisation='trace')
    estimator.fit(a)
    covariance = estimator.covariance_
    np_test.assert_equal(np.real(np.trace(covariance)), n_features)

    estimator = TylerShapeMatrix(tol=1e-4, iter_max=100,
                                 normalisation='determinant')
    estimator.fit(a)
    covariance = estimator.covariance_
    np_test.assert_almost_equal(
                        np.real(la.det(covariance)), 1)

    estimator = TylerShapeMatrix(tol=1e-4, iter_max=100,
                                 normalisation='element')
    estimator.fit(a)
    covariance = estimator.covariance_
    np_test.assert_almost_equal(covariance[0, 0], 1)

    n_features = 3
    n_samples = 10000*n_features
    covariance = generate_complex_covariance(n_features, unit_det=True)
    X = sample_complex_normal_distribution(n_samples, covariance).T
    estimator = TylerShapeMatrix(tol=1e-15, iter_max=100000,
                                 normalisation='determinant')
    estimator.fit(X)
    np_test.assert_array_almost_equal(estimator.covariance_,
                                      covariance, decimal=1)
