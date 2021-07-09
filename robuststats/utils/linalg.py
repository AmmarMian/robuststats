'''
File: linalg.py
File Created: Thursday, 8th July 2021 6:10:40 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Friday, 9th July 2021 10:59:34 am
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import numpy as np
from scipy.linalg import toeplitz


def hermitian(X):
    return .5*(X + X.conj().T)


def ToeplitzMatrix(rho, p, dtype=complex):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
                * dtype = dtype of array
            Outputs:
                * the matrix """

    return toeplitz(np.power(rho, np.arange(0, p))).astype(dtype)
