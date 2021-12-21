'''
File: test_centered_covariance.py
File Created: Tuesday, 21st December 2021 12:25:27 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 21st December 2021 5:10:05 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import unittest
import numpy as np
from robuststats.estimation.covariance import TylerShapeMatrix,\
    ComplexTylerShapeMatrix
from robuststats.models.mappings import check_Hermitian, check_Symmetric
from robuststats.models.probability import complex_multivariate_normal
from robuststats.utils.generation_data import generate_covariance, generate_complex_covariance
from scipy.stats import multivariate_normal
import numpy.testing as np_test


class TestTylerShapeMatrix(unittest.TestCase):
    seed = 77778
    np.random.seed(seed)
    n_features = 40
    n_samples = 400
    data = np.random.randn(n_samples, n_features)

    def test_fixed_point_shape(self):
        """Test class TylerShapeMatrix shape with fixed-point method.
        """
        estimator = TylerShapeMatrix()
        cov = estimator.fit_transform(self.data)
        assert np.isrealobj(cov)
        assert cov.shape == (self.n_features, self.n_features)
        assert check_Symmetric(cov)
        
    def test_natural_gradient(self):
        """Test class TylerShapeMatrix shape with natural gradient method.
        """
        estimator = TylerShapeMatrix(method="natural gradient")
        cov = estimator.fit_transform(self.data)
        assert np.isrealobj(cov)
        assert cov.shape == (self.n_features, self.n_features)
        assert check_Symmetric(cov)
        
    def test_natural_gradient_normalisation(self):
        """Test class TylerShapeMatrix normalisation with natural gradient method.
        """
        estimator = TylerShapeMatrix(method="natural gradient", normalisation="trace")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(np.trace(cov), self.n_features)
        
        estimator = TylerShapeMatrix(method="natural gradient", normalisation="determinant")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(np.linalg.det(cov), 1)
        
        estimator = TylerShapeMatrix(method="natural gradient", normalisation="element")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(cov[0,0], 1)

    def test_fixed_point_normalisation(self):
        """Test class TylerShapeMatrix normalisation with fixed-point method.
        """
        estimator = TylerShapeMatrix(normalisation="trace")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(np.trace(cov), self.n_features)
        
        estimator = TylerShapeMatrix(normalisation="determinant")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(np.linalg.det(cov), 1)
        
        estimator = TylerShapeMatrix(normalisation="element")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(cov[0,0], 1)
        
    def test_consistency(self):
        covariance = generate_covariance(3, unit_det=True)
        estimator = TylerShapeMatrix(normalisation="determinant")
        X = multivariate_normal(
            cov=covariance
            ).rvs(size=10000, random_state=self.seed)
        estimated_cov = estimator.fit_transform(X)
        np_test.assert_almost_equal(estimated_cov, covariance, decimal=1)
        


class TestComplexTylerShapeMatrix(unittest.TestCase):
    seed = 7779
    np.random.seed(seed)
    n_features = 40
    n_samples = 400
    data = np.random.randn(n_samples, n_features) + 1j*np.random.randn(n_samples, n_features)

    def test_fixed_point_shape(self):
        """Test class ComplexTylerShapeMatrix shape with fixed-point method.
        """
        estimator = ComplexTylerShapeMatrix()
        cov = estimator.fit_transform(self.data)
        assert np.iscomplexobj(cov)
        assert cov.shape == (self.n_features, self.n_features)
        assert check_Hermitian(cov)
        
    def test_natural_gradient(self):
        """Test class ComplexTylerShapeMatrix shape with natural gradient method.
        """
        estimator = ComplexTylerShapeMatrix(method="natural gradient")
        cov = estimator.fit_transform(self.data)
        assert np.iscomplexobj(cov)
        assert cov.shape == (self.n_features, self.n_features)
        assert check_Hermitian(cov)
        
    def test_natural_gradient_normalisation(self):
        """Test class ComplexTylerShapeMatrix normalisation with natural gradient method.
        """
        estimator = ComplexTylerShapeMatrix(method="natural gradient", normalisation="trace")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(np.trace(cov), self.n_features)
        
        estimator = ComplexTylerShapeMatrix(method="natural gradient", normalisation="determinant")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(np.linalg.det(cov), 1)
        
        estimator = ComplexTylerShapeMatrix(method="natural gradient", normalisation="element")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(cov[0,0], 1)

    def test_fixed_point_normalisation(self):
        """Test class ComplexTylerShapeMatrix normalisation with fixed-point method.
        """
        estimator = ComplexTylerShapeMatrix(normalisation="trace")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(np.trace(cov), self.n_features)
        
        estimator = ComplexTylerShapeMatrix(normalisation="determinant")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(np.linalg.det(cov), 1)
        
        estimator = ComplexTylerShapeMatrix(normalisation="element")
        cov = estimator.fit_transform(self.data)
        np_test.assert_almost_equal(cov[0,0], 1)
        
    def test_consistency(self):
        covariance = generate_complex_covariance(3, unit_det=True)
        estimator = ComplexTylerShapeMatrix(normalisation="determinant")
        X = complex_multivariate_normal(
            cov=covariance, random_state=self.seed
            ).rvs(size=10000)
        estimated_cov = estimator.fit_transform(X)
        np_test.assert_almost_equal(estimated_cov, covariance, decimal=1)