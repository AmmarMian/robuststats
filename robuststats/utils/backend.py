'''
File: backend.py
File Created: Tuesday, 30th November 2021 11:23:08 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 30th November 2021 12:12:42 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Functions to swith backends.
-----
Copyright 2021, Université Savoie Mont-Blanc
'''
import numpy as np
import autograd.numpy as np_a
import numpy.linalg as la
import autograd.numpy.linalg as la_a

def get_be_autograd(autograd):
    """Get backend between numpy or autograd.numpy

    Parameters
    ----------
    autograd : bool
        If true, backend is autograd.numpy. Otherwhise it
        is numpy.

    Returns
    -------
    module
        Either numpy or autograd.numpy
    """
    if autograd:
        be = np_a
    else:
        be = np
    return be


def get_be_la_autograd(autograd):
    """Get linalg backend between numpy or autograd.numpy

    Parameters
    ----------
    autograd : bool
        If true, backend is autograd.numpy.linalg. 
        Otherwhise it is numpy.linalg.

    Returns
    -------
    module
        Either numpy.linalg or autograd.numpy.linalg
    """
    if autograd:
        be_la = la_a
    else:
        be_la = la
    return be_la