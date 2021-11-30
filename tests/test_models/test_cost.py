'''
File: test_cost.py
File Created: Tuesday, 30th November 2021 11:33:51 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 30th November 2021 12:23:51 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import unittest
import numpy as np
import autograd

from robuststats.models.cost import Tyler_cost_real
from robuststats.utils.generation_data import generate_covariance

class TestTylercostNumpy(unittest.TestCase):
    np.random.seed(77)
    autograd = False
    n_features = 70
    n_samples = 200
    X = np.random.randn(n_samples, n_features)
    Q = generate_covariance(n_features)
    cost_value = Tyler_cost_real(X, Q, autograd)
    
    def test_Tyler_cost_is_scalar(self):
        assert np.isscalar(self.cost_value)
    def test_Tyler_cost_is_real(self):
        assert np.isrealobj(self.cost_value)


class TestTylercostAutograd(unittest.TestCase):
    np.random.seed(79)
    autograd = True
    n_features = 70
    n_samples = 200
    X = np.random.randn(n_samples, n_features)
    Q = generate_covariance(n_features)
    cost_value = Tyler_cost_real(X, Q, autograd)
    
    def test_Tyler_cost_is_scalar(self):
        assert np.isscalar(self.cost_value)
    def test_Tyler_cost_is_real(self):
        assert np.isrealobj(self.cost_value)    
    def test_Tyler_cost_is_differentiable(self):
        def cost(Q):
            return Tyler_cost_real(self.X, Q, self.autograd)
        grad = autograd.grad(cost, argnum=[0])(self.Q)
        assert grad[0].shape == (self.n_features, self.n_features)
        assert np.isrealobj(grad)

