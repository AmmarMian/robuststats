'''
File: verbose.py
File Created: Saturday, 19th June 2021 4:06:13 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Sunday, 20th June 2021 9:52:41 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''


import logging
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


class logging_tqdm(tqdm):
    def __init__(
            self,
            *args,
            logger: logging.Logger = None,
            mininterval: float = 1,
            bar_format: str = '{desc}{percentage:3.0f}%{r_bar}',
            desc: str = 'progress: ',
            **kwargs):
        self._logger = logger
        super().__init__(
            *args,
            mininterval=mininterval,
            bar_format=bar_format,
            desc=desc,
            **kwargs
        )

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return LOGGER

    def display(self, msg=None, pos=None):
        if not self.n:
            # skip progress bar before having processed anything
            return
        if not msg:
            msg = self.__str__()
        self.logger.info('%s', msg)


def matprint(mat, fmt="g"):
    """ Pretty print a matrix in Python 3 with numpy.
    Source: https://gist.github.com/lbn/836313e283f5d47d2e4e
    """

    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col])
                 for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
