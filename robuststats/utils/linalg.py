'''
File: linalg.py
File Created: Thursday, 8th July 2021 6:10:40 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 8th July 2021 6:11:08 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

def hermitian(X):
    return .5*(X + X.conj().T)