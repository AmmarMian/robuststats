'''
File: test_manifolds.py
Created Date: Saturday June 19th 2021 - 06:42pm
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Sat Jun 19 2021
Modified By: Ammar Mian
-----
Copyright (c) 2021 Université Savoie Mont-Blanc
'''

from robuststats.models.manifolds.structuredcovariance import (
                            KroneckerHermitianPositiveElliptical
                        )
import numpy.random as rnd
import numpy.testing as np_test
import numpy as np
from numpy import linalg as la
from scipy.linalg import expm

import logging
logging.basicConfig(level='INFO')


# TODO: tester avec k!=1
def test_inner():
    seed = 761
    rnd.seed(seed)

    a = np.random.randint(low=1, high=10)
    b = np.random.randint(low=1, high=20)
    d = np.random.randint(low=1, high=200)
    n = a*b
    alpha = (d+n)/(d+n+1)
    manifold = KroneckerHermitianPositiveElliptical(n, a, b, alpha)
    A, B = manifold.rand()
    theta = (A, B)
    xi_A, xi_B = manifold.randvec(theta)
    eta_A, eta_B = manifold.randvec(theta)

    desired_inner = np.real(alpha*b*np.trace(la.inv(A)@xi_A@la.inv(A)@eta_A) +
                            alpha*a*np.trace(la.inv(B)@xi_B@la.inv(B)@eta_B) +
                            (alpha-1)*(a**2)*np.trace(la.inv(B)@xi_B) *
                            np.trace(la.inv(B)@eta_B))
    inner = manifold.inner(theta, (xi_A, xi_B), (eta_A, eta_B))
    assert type(inner) is np.float64
    np_test.assert_almost_equal(inner, desired_inner)


def test_exp():
    seed = 76
    rnd.seed(seed)

    a = np.random.randint(low=1, high=10)
    b = np.random.randint(low=1, high=20)
    d = np.random.randint(low=1, high=200)
    n = a*b
    alpha = (d+n)/(d+n+1)
    manifold = KroneckerHermitianPositiveElliptical(n, a, b, alpha)
    A, B = manifold.rand()
    theta = (A, B)
    xi_A, xi_B = manifold.randvec(theta)

    desired_exp = (A@expm(la.inv(A)@xi_A), B@expm(la.inv(B)@xi_B))
    exp = manifold.exp(theta, (xi_A, xi_B))
    assert exp[0].shape == (a, a)
    assert exp[1].shape == (b, b)
    np_test.assert_almost_equal(exp[0], desired_exp[0])
    np_test.assert_almost_equal(exp[1], desired_exp[1])


def test_retr():
    seed = 761
    rnd.seed(seed)

    a = np.random.randint(low=1, high=10)
    b = np.random.randint(low=1, high=20)
    d = np.random.randint(low=1, high=200)
    n = a*b
    alpha = (d+n)/(d+n+1)

    manifold = KroneckerHermitianPositiveElliptical(n, a, b, alpha)
    A, B = manifold.rand()
    theta = (A, B)
    xi_A, xi_B = manifold.randvec(theta)

    desired_retr = (A + xi_A + (1/2)*xi_A@la.inv(A)@xi_A,
                    B + xi_B + (1/2)*xi_B@la.inv(B)@xi_B)
    retr = manifold.retr(theta, (xi_A, xi_B))
    assert retr[0].shape == (a, a)
    assert retr[1].shape == (b, b)
    np_test.assert_almost_equal(desired_retr[0], retr[0])
    np_test.assert_almost_equal(desired_retr[1], retr[1])
