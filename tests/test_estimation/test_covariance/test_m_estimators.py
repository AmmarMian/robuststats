'''
File: test_m_estimators.py
File Created: Tuesday, 21st December 2021 12:00:04 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 21st December 2021 4:49:12 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import unittest
import numpy as np
from robuststats.estimation.covariance._m_estimators import (
    _tyler_m_estimator_function, _huber_m_estimator_function,
    fixed_point_m_estimation_centered
)
from robuststats.models.mappings import check_Hermitian, check_Symmetric


class TestMEstimatorsFunctions(unittest.TestCase):
    seed = 7777
    np.random.seed(seed)
    n_samples = 1000
    data = 100+np.random.randn(n_samples)

    def test_huber_function(self):
        """Test M-estimating function of Huber.
        """
        quadratic_scaled = _huber_m_estimator_function(self.data)
        assert isinstance(quadratic_scaled, np.ndarray)
        assert quadratic_scaled.shape == (self.n_samples,)
        assert np.all(quadratic_scaled>=0)
        
        try:
            quadratic_scaled = _huber_m_estimator_function(self.data, lbda=0)
        except AssertionError:
            pass
        else:
            raise AssertionError("Function _huber_m_estimator_function should raise an error but doesn't")
        
    
    def test_tyler_function(self):
        """Test M-estimating function of Tyler.
        """
        quadratic_scaled = _tyler_m_estimator_function(self.data, n_features=45)
        assert isinstance(quadratic_scaled, np.ndarray)
        assert quadratic_scaled.shape == (self.n_samples,)
        assert np.all(quadratic_scaled>=0)


class TestFixedPointEstimator(unittest.TestCase):
    seed = 777
    np.random.seed(seed)
    n_features = 40
    n_samples = 100
    data_real = np.random.randn(n_samples, n_features)
    data_complex = data_real + 1j*np.random.randn(n_samples, n_features)
    
    def test_tyler_real(self):
        """Testing fixed-point algorithm with Tyler function real-valued.
        """
        cov, err, iteration = fixed_point_m_estimation_centered(
            self.data_real, _tyler_m_estimator_function, n_features=self.n_features
        )
        assert np.isrealobj(cov)
        assert cov.shape == (self.n_features, self.n_features)
        assert check_Symmetric(cov)
    
    def test_tyler_complex(self):
        """Testing fixed-point algorithm with Tyler function complex-valued.
        """
        cov, err, iteration = fixed_point_m_estimation_centered(
            self.data_complex, _tyler_m_estimator_function, n_features=self.n_features
        )
        assert np.iscomplexobj(cov)
        assert cov.shape == (self.n_features, self.n_features) 
        assert check_Hermitian(cov)  
        
    def test_huber_real(self):
        """Testing fixed-point algorithm with Huber function real-valued.
        """
        cov, err, iteration = fixed_point_m_estimation_centered(
            self.data_real, _huber_m_estimator_function, n_features=self.n_features
        )
        assert np.isrealobj(cov)
        assert cov.shape == (self.n_features, self.n_features)
        assert check_Symmetric(cov)
    
    def test_huber_complex(self):
        """Testing fixed-point algorithm with Huber function complex-valued.
        """
        cov, err, iteration = fixed_point_m_estimation_centered(
            self.data_complex, _huber_m_estimator_function, n_features=self.n_features
        )
        assert np.iscomplexobj(cov)
        assert cov.shape == (self.n_features, self.n_features)
        assert check_Hermitian(cov)
