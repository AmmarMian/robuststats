'''
File: test_mappings.py
Created Date: Friday June 18th 2021 - 10:23pm
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Thursday, 18th November 2021 4:19:49 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright (c) 2021 Universit√© Savoie Mont-Blanc
'''
from robuststats.models.mappings import check_Hermitian, iscovariance,\
            covariancetoreal, covariancetocomplex, covariancestoreal, \
            covariancestocomplex, arraytoreal, arraytocomplex
from robuststats.utils.generation_data import generate_complex_covariance
# from robuststats.utils.verbose import matprint
import numpy.random as rnd
import numpy.testing as np_test
import numpy as np

import logging
logging.basicConfig(level='INFO')


def test_check_Hermitian():
    seed = 761
    rnd.seed(seed)

    a = np.random.randn(17, 17) + 1j*np.random.randn(17, 17)
    a_hermitian = (a + a.T.conj())/2

    assert check_Hermitian(a_hermitian)
    assert check_Hermitian(a) is False


def test_iscovariance():
    seed = 761
    rnd.seed(seed)

    a = generate_complex_covariance(17)
    b = np.random.randn(17, 17, 8)
    c = np.random.randn(17, 5)

    assert iscovariance(a)
    assert iscovariance(b) is False
    assert iscovariance(c) is False


def test_covariancetoreal():
    seed = 761
    rnd.seed(seed)

    n_features = 17
    a = generate_complex_covariance(n_features)
    a_real = covariancetoreal(a)
    assert a_real.ndim == 2
    assert a_real.shape == (2*n_features, 2*n_features)
    assert np.isrealobj(a_real)
    np_test.assert_equal(covariancetoreal(a_real), a_real)


def test_covariancetocomplex():
    seed = 761
    rnd.seed(seed)

    n_features = 17
    a = generate_complex_covariance(n_features)
    a_real = covariancetoreal(a)
    a_bis = covariancetocomplex(a_real)

    assert a_bis.ndim == 2
    assert a_bis.shape == (n_features, n_features)
    assert np.iscomplexobj(a_bis)
    np_test.assert_equal(a, a_bis)
    np_test.assert_equal(covariancetocomplex(a), a)


def test_covariancestoreal():
    seed = 761
    rnd.seed(seed)

    n_features = 17
    n_samples = 100
    a = np.zeros((n_samples, n_features, n_features), dtype=complex)
    for i in range(n_samples):
        a[i] = generate_complex_covariance(n_features)
    a_real = covariancestoreal(a)

    assert a_real.ndim == 3
    assert a_real.shape == (100, 2*n_features, 2*n_features)
    assert np.isrealobj(a_real)
    np_test.assert_equal(covariancestoreal(a_real), a_real)


def test_covariancestocomplex():
    seed = 761
    rnd.seed(seed)

    n_features = 17
    n_samples = 100
    a = np.zeros((n_samples, n_features, n_features), dtype=complex)
    for i in range(n_samples):
        a[i] = generate_complex_covariance(n_features)
    a_real = covariancestoreal(a)
    a_bis = covariancestocomplex(a_real)
    assert a_bis.ndim == 3
    assert a_bis.shape == (100, n_features, n_features)
    assert np.iscomplexobj(a_bis)
    np_test.assert_equal(a, a_bis)
    np_test.assert_equal(covariancestocomplex(a), a)


def test_arraytoreal():
    seed = 761
    rnd.seed(seed)

    n_features = 17
    n_samples = 200

    # 1 axis case
    a = np.random.randn(n_features) + 1j*np.random.randn(n_features)
    a_real = arraytoreal(a)
    assert a_real.ndim == 1
    assert len(a_real) == 2*n_features
    assert np.isrealobj(a_real)

    # 2 axis case
    a = np.random.randn(n_samples, n_features) + \
        1j*np.random.randn(n_samples, n_features)
    a_real = arraytoreal(a)

    assert a_real.ndim == 2
    assert a_real.shape == (n_samples, 2*n_features)
    assert np.isrealobj(a_real)


def test_arraytocomplex():
    seed = 761
    rnd.seed(seed)

    n_features = 17
    n_samples = 200

    # 1 axis case
    a = np.random.randn(n_features) + 1j*np.random.randn(n_features)
    a_real = arraytoreal(a)
    a_bis = arraytocomplex(a_real)
    assert a_bis.ndim == 1
    assert len(a_bis) == n_features
    assert np.iscomplexobj(a_bis)

    # 2 axis case
    a = np.random.randn(n_samples, n_features) + \
        1j*np.random.randn(n_samples, n_features)
    a_real = arraytoreal(a)
    a_bis = arraytocomplex(a_real)

    assert a_bis.ndim == 2
    assert a_bis.shape == (n_samples, n_features)
    assert np.iscomplexobj(a_bis)
