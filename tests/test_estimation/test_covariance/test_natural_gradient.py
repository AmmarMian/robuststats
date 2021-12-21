'''
File: test_natural_gradient.py
File Created: Tuesday, 21st December 2021 12:46:06 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 21st December 2021 4:46:18 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import unittest
import numpy as np
from robuststats.estimation.covariance._natural_gradient import *
from robuststats.models.mappings import check_Hermitian, check_Symmetric
