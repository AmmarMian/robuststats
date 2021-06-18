'''
File: montecarlo.py
Created Date: Friday April 23rd 2021 - 11:41am
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Fri Apr 23 2021
Modified By: Ammar Mian
-----
Copyright (c) 2021 Université Savoie Mont-Blanc
'''
import contextlib
import numpy as np

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)