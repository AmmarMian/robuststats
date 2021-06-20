'''
File: test_base.py
File Created: Sunday, 20th June 2021 7:11:16 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Sunday, 20th June 2021 7:29:50 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

from robuststats.models.mappings.complexreal import check_Hermitian
from robuststats.estimation.base import ComplexEmpiricalCovariance,\
    empirical_covariance
from pyCovariance.generation_data import generate_complex_covariance

import numpy.random as rnd
import numpy.testing as np_test
import numpy as np

import logging
logging.basicConfig(level='INFO')


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
    covariace_from_function = empirical_covariance(a)

    assert np.iscomplexobj(covariance)
    assert covariance.shape == (n_features, n_features)
    assert check_Hermitian(covariance)
    assert check_Hermitian(covariace_from_function)
    np_test.assert_array_equal(covariace_from_function, covariance)
    assert estimator.error_norm(covariance) == 0
    assert estimator.error_norm(covariance + np.eye(n_features)) >= 0
    estimator._set_covariance(covariance + np.eye(n_features))
    np_test.assert_array_equal(covariance + np.eye(n_features),
                               estimator.covariance_)
