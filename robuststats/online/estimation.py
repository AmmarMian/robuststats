'''
File: estimation.py
File Created: Thursday, 8th July 2021 4:01:38 pm
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Thursday, 8th July 2021 5:56:06 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

from robuststats.estimation.base import ComplexEmpiricalCovariance
import logging
from tqdm import tqdm


class StochasticGradientCovarianceEstimator(ComplexEmpiricalCovariance):
    def __init__(self, manifold, rgrad, lrate,
                       estimatetocov=None,
                       cost=None, verbosity=False):
        super().__init__()
        self._manifold = manifold
        self._rgrad = rgrad
        self.lrate_ = lrate
        self._cost = cost
        self._verbosity = verbosity
        self._estimatetocov = estimatetocov

        self._iteration = 0
        self.estimate_ = None

    def _estimate_to_cov(self, estimate):
        if self._estimatetocov is None:
            return estimate
        else:
            return self._estimatetocov(estimate)

    def set_estimate(self, estimate):
        self.estimate_ = estimate
        covariance = self._estimate_to_cov(estimate)
        self._set_covariance(covariance)

    def _current_lr(self, lr_update, iteration):
        if lr_update == "inverse iteration":
            return self.lrate_/(iteration+1)
        else:
            return self.lrate_

    def update(self, X, lr_update):

        if self.estimate_ is None:
            logging.error(
             "Estimator not initilaized, use method set_estimate beforehand."
            )
            return self

        # TODO: constant step + geodesic between past two iterations
        nabla = self._rgrad(X, self.estimate_)
        t = self._current_lr(lr_update, self._iteration)
        if isinstance(nabla, list):
            t_nabla = [t*x for x in nabla]
        elif isinstance(nabla, tuple):
            t_nabla = tuple([t*x for x in nabla])
        else:
            t_nabla = t*nabla

        estimate = self._manifold.retr(self.estimate_, t_nabla)
        self.set_estimate(estimate)
        return self

    def fit(self, X, y=None, init=None, lr_update="inverse iteration"):

        self.set_estimate(init)
        if self.estimate_ is None:
            logging.error(
             "Estimator not initilaized. Please provide an initial point."
            )
            return self

        n_samples, n_features = X.shape
        if self._verbosity:
            pbar_v = tqdm(total=n_samples)

        for iteration in range(n_samples):
            self.update(X[iteration, :].reshape((1, n_features)), lr_update)

            if self._verbosity:
                pbar_v.update()
                if self._cost is not None:
                    cost = self._cost(X, self.estimate_)
                    pbar_v.set_description(f'(err={cost})', refresh=True)

            self._iteration += 1

        return self
