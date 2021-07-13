'''
File: test_estimation.py
Created Date: Monday July 12th 2021 - 01:59pm
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Mon Jul 12 2021
Modified By: Ammar Mian
-----
Copyright (c) 2021 Université Savoie Mont-Blanc
'''


from robuststats.models.mappings import check_Hermitian
from robuststats.online.estimation import StochasticGradientCovarianceEstimator
from robuststats.models.probability import complex_multivariate_normal
from robuststats.utils.linalg import ToeplitzMatrix
from pymanopt.manifolds.hpd import HermitianPositiveDefinite
from pyCovariance.generation_data import generate_complex_covariance,\
    sample_complex_normal_distribution

import numpy.random as rnd
import numpy.testing as np_test
import numpy as np
import numpy.linalg as la

import logging
logging.basicConfig(level='INFO')
