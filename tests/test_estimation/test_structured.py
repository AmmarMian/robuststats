'''
File: test_structured.py
File Created: Thursday, 8th July 2021 1:23:20 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tue Jul 13 2021
Modified By: Ammar Mian
-----
Copyright 2021, Université Savoie Mont-Blanc
'''


from robuststats.estimation.structured import KroneckerStudent,\
    _generate_egrad_function_kronecker_student, _generate_cost_function_kronecker_student, KroneckerEllipticalMM,\
    _estimate_covariance_kronecker_t_gradient, estimation_cov_kronecker_MM
from robuststats.models.manifolds import KroneckerHermitianPositiveElliptical
from robuststats.models.probability import complex_multivariate_t
from robuststats.online.estimation import StochasticGradientCovarianceEstimator
from robuststats.utils.linalg import hermitian

import numpy.random as rnd
import numpy.testing as np_test
import numpy as np
import numpy.linalg as la

import logging
logging.basicConfig(level='INFO')


def test_estimation_cov_kronecker_MM():
    seed = 760
    rnd.seed(seed)

    a = 7
    b = 9
    n_features = a*b
    df = 3
    alpha = (df+n_features)/(df+n_features+1)
    manifold = KroneckerHermitianPositiveElliptical(a, b, alpha)
    A, B = manifold.rand()
    covariance = np.kron(A, B)

    n_samples = 10000
    model = complex_multivariate_t(shape=covariance, df=df)
    X = model.rvs(size=n_samples)

    A, B, _, iteration = estimation_cov_kronecker_MM(X, a, b, iter_max=100)

    assert np.iscomplexobj(A)
    assert np.iscomplexobj(B)
    assert A.shape == (a, a)
    assert B.shape == (b, b)
    assert iteration <= 100


def test_KroneckerEllipticalMM():
    seed = 761
    rnd.seed(seed)

    a = 7
    b = 9
    n_features = a*b
    df = 3
    alpha = (df+n_features)/(df+n_features+1)
    manifold = KroneckerHermitianPositiveElliptical(a, b, alpha)
    A, B = manifold.rand()
    covariance = np.kron(A, B)

    n_samples = 10000
    model = complex_multivariate_t(shape=covariance, df=df)
    X = model.rvs(size=n_samples)

    estimator = KroneckerEllipticalMM(a, b, iter_max=100)
    estimator.fit(X)

    assert np.iscomplexobj(estimator.A_)
    assert np.iscomplexobj(estimator.B_)
    assert estimator.A_.shape == (a, a)
    assert estimator.B_.shape == (b, b)


def test__generate_cost_function_kronecker_student():
    seed = 762
    rnd.seed(seed)

    a = 7
    b = 9
    n_features = a*b
    df = 3
    alpha = (df+n_features)/(df+n_features+1)
    manifold = KroneckerHermitianPositiveElliptical(a, b, alpha)
    A, B = manifold.rand()
    covariance = np.kron(A, B)

    n_samples = 10000
    model = complex_multivariate_t(shape=covariance, df=df)
    X = model.rvs(size=n_samples)

    cost_function = _generate_cost_function_kronecker_student(X, df)
    assert np.isscalar(cost_function(A, B))


def test__generate_egrad_function_kronecker_student():
    seed = 763
    rnd.seed(seed)

    a = 7
    b = 9
    n_features = a*b
    df = 3
    alpha = (df+n_features)/(df+n_features+1)
    manifold = KroneckerHermitianPositiveElliptical(a, b, alpha)
    A, B = manifold.rand()
    covariance = np.kron(A, B)

    n_samples = 10000
    model = complex_multivariate_t(shape=covariance, df=df)
    X = model.rvs(size=n_samples)

    egrad_function = _generate_egrad_function_kronecker_student(X, a, b, df)
    grad_A, grad_B = egrad_function(A, B)
    assert np.iscomplexobj(grad_A)
    assert np.iscomplexobj(grad_B)
    assert grad_A.shape == (a, a)
    assert grad_B.shape == (b, b)


def test__estimate_covariance_kronecker_t_gradient():
    seed = 764
    rnd.seed(seed)

    a = 7
    b = 9
    n_features = a*b
    df = 3
    alpha = (df+n_features)/(df+n_features+1)
    manifold = KroneckerHermitianPositiveElliptical(a, b, alpha)
    A, B = manifold.rand()
    covariance = np.kron(A, B)

    n_samples = 10000
    model = complex_multivariate_t(shape=covariance, df=df)
    X = model.rvs(size=n_samples)

    A, B = _estimate_covariance_kronecker_t_gradient(X, df, a, b, manifold)
    assert np.iscomplexobj(A)
    assert np.iscomplexobj(B)
    assert A.shape == (a, a)
    assert B.shape == (b, b)


def test_KroneckerStudent():
    seed = 765
    rnd.seed(seed)

    a = 7
    b = 9
    n_features = a*b
    df = 3
    alpha = (df+n_features)/(df+n_features+1)
    manifold = KroneckerHermitianPositiveElliptical(a, b, alpha)
    A, B = manifold.rand()
    covariance = np.kron(A, B)

    n_samples = 10000
    model = complex_multivariate_t(shape=covariance, df=df)
    X = model.rvs(size=n_samples)

    estimator = KroneckerStudent(a, b, df)
    estimator.fit(X)

    assert np.iscomplexobj(estimator.A_)
    assert np.iscomplexobj(estimator.B_)
    assert estimator.A_.shape == (a, a)
    assert estimator.B_.shape == (b, b)
