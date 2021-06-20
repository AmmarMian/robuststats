'''
File: kronecker.py
File Created: Sunday, 20th June 2021 4:47:02 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Sunday, 20th June 2021 5:31:15 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import numpy as np
import numpy.linalg as la
import logging

from sklearn.covariance import EmpiricalCovariance

class KroneckerElliptical(EmpiricalCovariance):
    pass