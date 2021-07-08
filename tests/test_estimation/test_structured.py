'''
File: test_structured.py
File Created: Thursday, 8th July 2021 1:23:20 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 8th July 2021 2:49:21 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''


from robuststats.models.mappings import check_Hermitian
from robuststats.estimation.structured import KroneckerStudent
from pyCovariance.generation_data import generate_complex_covariance,\
    sample_complex_normal_distribution

import numpy.random as rnd
import numpy.testing as np_test
import numpy as np
import numpy.linalg as la

import logging
logging.basicConfig(level='INFO')