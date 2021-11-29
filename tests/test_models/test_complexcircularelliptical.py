'''
File: test_complexcircularelliptical.py
File Created: Monday, 21st June 2021 1:58:36 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Monday, 29th November 2021 12:03:12 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''


from robuststats.models.probability import complex_multivariate_normal_frozen,\
    complex_multivariate_normal, _check_parameters_array_complex_normal,\
    _check_parameters_complex_normal, complex_multivariate_t_frozen,\
    complex_multivariate_t
from robuststats.models.mappings import check_Hermitian
from robuststats.utils.generation_data import generate_complex_covariance,\
    generate_covariance
import unittest
# from robuststats.utils.verbose import matprint
import numpy.random as rnd
import numpy.testing as np_test
import numpy as np
import numpy.linalg as la
from scipy.special import loggamma

import logging
logging.basicConfig(level='INFO')

# TODO : make unittest classes
# class TestParameters(unittest.TestCase):
#     seed = 761
#     n_features = 17
#     def test__check_parameters_complex_normal_1(self):
#         """Test fonction _check_parameters_complex_normal.
#         Case mean is None and cov real
#         """
#         mean = None
#         covariance = generate_covariance(self.n_features)
#         mean, covariance = _check_parameters_complex_normal(mean, covariance)
#         assert np.isrealobj(mean)
#         assert np.isrealobj(covariance)
#         assert mean.shape == (self.n_features,)
#         assert covariance.shape == (self.n_features, self.n_features)
#         assert check_Hermitian(covariance)

def test__check_parameters_complex_normal():
    seed = 761
    rnd.seed(seed)

    n_features = 17

    # Case mean is None and cov real
    mean = None
    covariance = generate_covariance(n_features)
    mean, covariance = _check_parameters_complex_normal(mean, covariance)
    assert np.isrealobj(mean)
    assert np.isrealobj(covariance)
    assert mean.shape == (n_features,)
    assert covariance.shape == (n_features, n_features)
    assert check_Hermitian(covariance)

    # Case mean is None and cov complex
    mean = None
    covariance = generate_complex_covariance(n_features)
    mean, covariance = _check_parameters_complex_normal(mean, covariance)
    assert np.isrealobj(mean)
    assert np.isrealobj(covariance)
    assert mean.shape == (2*n_features,)
    assert covariance.shape == (2*n_features, 2*n_features)
    assert check_Hermitian(covariance)

    # Case both real
    mean = np.zeros((n_features,))
    covariance = generate_covariance(n_features)
    mean, covariance = _check_parameters_complex_normal(mean, covariance)
    assert np.isrealobj(mean)
    assert np.isrealobj(covariance)
    assert mean.shape == (n_features,)
    assert covariance.shape == (n_features, n_features)
    assert check_Hermitian(covariance)

    # Case one real one complex 1/2
    mean = np.zeros((n_features,))
    covariance = generate_complex_covariance(n_features)
    mean, covariance = _check_parameters_complex_normal(mean, covariance)
    assert np.isrealobj(mean)
    assert np.isrealobj(covariance)
    assert mean.shape == (2*n_features,)
    assert covariance.shape == (2*n_features, 2*n_features)
    assert check_Hermitian(covariance)

    # Case one real one complex 2/2
    mean = np.zeros((n_features,), dtype=complex)
    covariance = generate_covariance(n_features)
    mean, covariance = _check_parameters_complex_normal(mean, covariance)
    assert np.isrealobj(mean)
    assert np.isrealobj(covariance)
    assert mean.shape == (2*n_features,)
    assert covariance.shape == (2*n_features, 2*n_features)
    assert check_Hermitian(covariance)

    # Case both complex
    mean = np.zeros((n_features,), dtype=complex)
    covariance = generate_complex_covariance(n_features)
    mean, covariance = _check_parameters_complex_normal(mean, covariance)
    assert np.isrealobj(mean)
    assert np.isrealobj(covariance)
    assert mean.shape == (2*n_features,)
    assert covariance.shape == (2*n_features, 2*n_features)
    assert check_Hermitian(covariance)


def test__check_parameters_array_complex_normal():
    seed = 761
    rnd.seed(seed)

    n_features = 17
    n_samples = 1000

    # Case mean is None and cov complex, x complex
    mean = None
    covariance = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features) +1j
    x_map, mean_map, cov_map = _check_parameters_array_complex_normal(
                                                x, mean, covariance)
    assert np.isrealobj(x_map)
    assert x_map.shape == (n_samples, 2*n_features)
    assert mean_map.shape == (2*n_features,)
    assert cov_map.shape == (2*n_features, 2*n_features)


    # Case mean is None and cov complex, x real
    mean = None
    covariance = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    x_map, mean_map, cov_map = _check_parameters_array_complex_normal(
                                                x, mean, covariance)
    assert np.isrealobj(x_map)
    assert x_map.shape == (n_samples, 2*n_features)
    assert mean_map.shape == (2*n_features,)
    assert cov_map.shape == (2*n_features, 2*n_features)


    # Case mean is None and cov real
    mean = None
    covariance = generate_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    x_map, mean_map, cov_map = _check_parameters_array_complex_normal(
                                                x, mean, covariance)
    assert np.isrealobj(x_map)
    np_test.assert_array_equal(x_map, x)
    np_test.assert_array_equal(mean_map, mean)
    np_test.assert_array_equal(cov_map, covariance)


    # Case all real
    mean = np.zeros((n_features,))
    covariance = generate_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    x_map, mean_map, cov_map = _check_parameters_array_complex_normal(
                                                x, mean, covariance)
    assert np.isrealobj(x_map)
    np_test.assert_array_equal(x_map, x)
    np_test.assert_array_equal(mean_map, mean)
    np_test.assert_array_equal(cov_map, covariance)

    # Case data real, parameters complex
    mean = np.zeros((n_features,), dtype=complex)
    covariance = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    x_map, mean_map, cov_map = _check_parameters_array_complex_normal(
                                                x, mean, covariance)
    assert np.isrealobj(x_map)
    assert x_map.shape == (n_samples, 2*n_features)
    assert mean_map.shape == (2*n_features,)
    assert cov_map.shape == (2*n_features, 2*n_features)

    # Case data complex, parameters real
    mean = np.zeros((n_features,))
    covariance = generate_covariance(n_features)
    x = np.random.randn(n_samples, n_features) + 1j
    x_map, mean_map, cov_map = _check_parameters_array_complex_normal(
                                                x, mean, covariance)
    assert np.isrealobj(x_map)
    assert x_map.shape == (n_samples, n_features)
    assert mean_map.shape == (n_features,)
    assert cov_map.shape == (n_features, n_features)

    # Case data complex, parameters complex
    mean = np.zeros((n_features,), dtype=complex)
    covariance = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features) + 1j
    x_map, mean_map, cov_map = _check_parameters_array_complex_normal(
                                                x, mean, covariance)
    assert np.isrealobj(x_map)
    assert x_map.shape == (n_samples, 2*n_features)
    assert mean_map.shape == (2*n_features,)
    assert cov_map.shape == (2*n_features, 2*n_features)


def test_complex_multivariate_normal_frozen():

    seed = 761
    rnd.seed(seed)

    n_features = 7
    n_samples = 10

    # Both data and parameters complex
    mean = np.zeros((n_features,), dtype=complex)
    covariance = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features) + 1j
    model = complex_multivariate_normal_frozen(mean=mean, cov=covariance)
    x_sampled = model.rvs(n_samples)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)
    assert model.logcdf(x).shape == (n_samples,)
    assert model.cdf(x).shape == (n_samples,)
    assert np.all(model.cdf(x) >= 0)
    assert np.all(model.cdf(x) <= 1)
    assert np.iscomplexobj(x_sampled)
    assert x_sampled.shape == (n_samples, n_features)

    # Testing if logpdf is accurate when complex data and parameters
    log_pdf = -np.ones((n_samples,), dtype=float)*n_features*np.log(np.pi) -\
        np.log(np.abs(la.det(covariance)))
    inv_covariance = la.inv(covariance)
    for k in range(n_samples):
        y = x_sampled[k, :].reshape((n_features, 1))
        log_pdf[k] -= np.trace(np.real(y @ y.T.conj() @ inv_covariance))
    np_test.assert_almost_equal(log_pdf, model.logpdf(x_sampled))

    # Both data and parameters real
    mean = np.zeros((n_features,))
    covariance = generate_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    model = complex_multivariate_normal_frozen(mean=mean, cov=covariance)
    x_sampled = model.rvs(n_samples)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)
    assert model.logcdf(x).shape == (n_samples,)
    assert model.cdf(x).shape == (n_samples,)
    assert np.all(model.cdf(x) >= 0)
    assert np.all(model.cdf(x) <= 1)
    assert np.isrealobj(x_sampled)
    assert x_sampled.shape == (n_samples, n_features)

    # data real and parameters complex
    mean = np.zeros((n_features,), dtype=complex)
    covariance = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    model = complex_multivariate_normal_frozen(mean=mean, cov=covariance)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)
    assert model.logcdf(x).shape == (n_samples,)
    assert model.cdf(x).shape == (n_samples,)
    assert np.all(model.cdf(x) >= 0)
    assert np.all(model.cdf(x) <= 1)

    # data complex and parameters real
    mean = np.zeros((n_features,))
    covariance = generate_covariance(n_features)
    x = np.random.randn(n_samples, n_features) + 1j
    model = complex_multivariate_normal_frozen(mean=mean, cov=covariance)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)
    assert model.logcdf(x).shape == (n_samples,)
    assert model.cdf(x).shape == (n_samples,)
    assert np.all(model.cdf(x) >= 0)
    assert np.all(model.cdf(x) <= 1)


def test_complex_multivariate_normal_call():
    # TODO: verify logpdf is that of the model with parameters mean
    # covariance chosen.
    seed = 761
    rnd.seed(seed)

    n_features = 7
    n_samples = 10

    # Both data and parameters complex
    mean = np.zeros((n_features,), dtype=complex)
    covariance = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features) + 1j
    model = complex_multivariate_normal(mean=mean, cov=covariance)
    x_sampled = model.rvs(n_samples)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)
    assert model.logcdf(x).shape == (n_samples,)
    assert model.cdf(x).shape == (n_samples,)
    assert np.all(model.cdf(x) >= 0)
    assert np.all(model.cdf(x) <= 1)
    assert np.iscomplexobj(x_sampled)
    assert x_sampled.shape == (n_samples, n_features)

    # Both data and parameters real
    mean = np.zeros((n_features,))
    covariance = generate_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    model = complex_multivariate_normal(mean=mean, cov=covariance)
    x_sampled = model.rvs(n_samples)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)
    assert model.logcdf(x).shape == (n_samples,)
    assert model.cdf(x).shape == (n_samples,)
    assert np.all(model.cdf(x) >= 0)
    assert np.all(model.cdf(x) <= 1)
    assert np.isrealobj(x_sampled)
    assert x_sampled.shape == (n_samples, n_features)

    # data real and parameters complex
    mean = np.zeros((n_features,), dtype=complex)
    covariance = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    model = complex_multivariate_normal(mean=mean, cov=covariance)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)
    assert model.logcdf(x).shape == (n_samples,)
    assert model.cdf(x).shape == (n_samples,)
    assert np.all(model.cdf(x) >= 0)
    assert np.all(model.cdf(x) <= 1)

    # data complex and parameters real
    mean = np.zeros((n_features,))
    covariance = generate_covariance(n_features)
    x = np.random.randn(n_samples, n_features) + 1j
    model = complex_multivariate_normal(mean=mean, cov=covariance)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)
    assert model.logcdf(x).shape == (n_samples,)
    assert model.cdf(x).shape == (n_samples,)
    assert np.all(model.cdf(x) >= 0)
    assert np.all(model.cdf(x) <= 1)


def test_complex_multivariate_normal():
    # TODO: verify logpdf is that of the model with parameters mean
    # covariance chosen.
    seed = 761
    rnd.seed(seed)

    n_features = 7
    n_samples = 10

    # Both data and parameters complex
    mean = np.zeros((n_features,), dtype=complex)
    covariance = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features) + 1j
    model = complex_multivariate_normal
    x_sampled = model.rvs(mean, covariance, n_samples)
    assert model.logpdf(x, mean, covariance).shape == (n_samples,)
    assert model.pdf(x, mean, covariance).shape == (n_samples,)
    assert np.all(model.pdf(x, mean, covariance) >= 0)
    assert np.all(model.pdf(x, mean, covariance) <= 1)
    assert model.logcdf(x, mean, covariance).shape == (n_samples,)
    assert model.cdf(x, mean, covariance).shape == (n_samples,)
    assert np.all(model.cdf(x, mean, covariance) >= 0)
    assert np.all(model.cdf(x, mean, covariance) <= 1)
    assert np.iscomplexobj(x_sampled)
    assert x_sampled.shape == (n_samples, n_features)

    # Both data and parameters real
    mean = np.zeros((n_features,))
    covariance = generate_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    model = complex_multivariate_normal
    x_sampled = model.rvs(mean, covariance, n_samples)
    assert model.logpdf(x, mean, covariance).shape == (n_samples,)
    assert model.pdf(x, mean, covariance).shape == (n_samples,)
    assert np.all(model.pdf(x, mean, covariance) >= 0)
    assert np.all(model.pdf(x, mean, covariance) <= 1)
    assert model.logcdf(x, mean, covariance).shape == (n_samples,)
    assert model.cdf(x, mean, covariance).shape == (n_samples,)
    assert np.all(model.cdf(x, mean, covariance) >= 0)
    assert np.all(model.cdf(x, mean, covariance) <= 1)
    assert np.isrealobj(x_sampled)
    assert x_sampled.shape == (n_samples, n_features)

    # data real and parameters complex
    mean = np.zeros((n_features,), dtype=complex)
    covariance = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    model = complex_multivariate_normal
    x_sampled = model.rvs(mean, covariance, n_samples)
    assert model.logpdf(x, mean, covariance).shape == (n_samples,)
    assert model.pdf(x, mean, covariance).shape == (n_samples,)
    assert np.all(model.pdf(x, mean, covariance) >= 0)
    assert np.all(model.pdf(x, mean, covariance) <= 1)
    assert model.logcdf(x, mean, covariance).shape == (n_samples,)
    assert model.cdf(x, mean, covariance).shape == (n_samples,)
    assert np.all(model.cdf(x, mean, covariance) >= 0)
    assert np.all(model.cdf(x, mean, covariance) <= 1)
    assert np.iscomplexobj(x_sampled)
    assert x_sampled.shape == (n_samples, n_features)

    # data complex and parameters real
    mean = np.zeros((n_features,))
    covariance = generate_covariance(n_features)
    x = np.random.randn(n_samples, n_features) + 1j
    model = complex_multivariate_normal
    x_sampled = model.rvs(mean, covariance, n_samples)
    assert model.logpdf(x, mean, covariance).shape == (n_samples,)
    assert model.pdf(x, mean, covariance).shape == (n_samples,)
    assert np.all(model.pdf(x, mean, covariance) >= 0)
    assert np.all(model.pdf(x, mean, covariance) <= 1)
    assert model.logcdf(x, mean, covariance).shape == (n_samples,)
    assert model.cdf(x, mean, covariance).shape == (n_samples,)
    assert np.all(model.cdf(x, mean, covariance) >= 0)
    assert np.all(model.cdf(x, mean, covariance) <= 1)
    assert np.isrealobj(x_sampled)
    assert x_sampled.shape == (n_samples, n_features)


def test_complex_multivariate_t_frozen():

    seed = 761
    rnd.seed(seed)

    n_features = 7
    n_samples = 10
    df = 3

    # Both data and parameters complex
    loc = np.zeros((n_features,), dtype=complex)
    shape = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features) + 1j
    model = complex_multivariate_t_frozen(loc=loc, shape=shape, df=df)
    x_sampled = model.rvs(n_samples)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)
    assert np.iscomplexobj(x_sampled)
    assert x_sampled.shape == (n_samples, n_features)

    # TODO: Testing if logpdf is accurate when complex data and parameters
    log_pdf = -np.ones((n_samples,), dtype=float)*n_features*np.log(np.pi) +\
        loggamma(df + n_features) - loggamma(df) - n_features*np.log(df) -\
        np.log(np.abs(la.det(shape)))
    inv_covariance = la.inv(shape)
    for k in range(n_samples):
        y = x_sampled[k, :].reshape((n_features, 1))
        log_pdf[k] -= (n_features+df)*np.log(
            1 + np.trace(np.real(y @ y.T.conj() @ inv_covariance)))
    # np_test.assert_almost_equal(log_pdf, model.logpdf(x_sampled))

    # Both data and parameters real
    loc = np.zeros((n_features,))
    shape = generate_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    model = complex_multivariate_t_frozen(loc=loc, shape=shape, df=df)
    x_sampled = model.rvs(n_samples)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)
    assert np.isrealobj(x_sampled)
    assert x_sampled.shape == (n_samples, n_features)

    # data real and parameters complex
    loc = np.zeros((n_features,), dtype=complex)
    shape = generate_complex_covariance(n_features)
    x = np.random.randn(n_samples, n_features)
    model = complex_multivariate_t_frozen(loc=loc, shape=shape, df=df)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)

    # data complex and parameters real
    loc = np.zeros((n_features,))
    shape = generate_covariance(n_features)
    x = np.random.randn(n_samples, n_features) + 1j
    model = complex_multivariate_t_frozen(loc=loc, shape=shape, df=df)
    assert model.logpdf(x).shape == (n_samples,)
    assert model.pdf(x).shape == (n_samples,)
    assert np.all(model.pdf(x) >= 0)
    assert np.all(model.pdf(x) <= 1)
