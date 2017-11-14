from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from filterpy.common import dot3
from filterpy.kalman import unscented_transform
from filterpy.stats import logpdf
import numpy as np
from numpy import eye, zeros, dot, isscalar, outer
from scipy.linalg import inv
from filterpy.kalman import UnscentedKalmanFilter


class EstimationUnscentedKalmanFilter(UnscentedKalmanFilter):
    """ Adapts the filterpy UKF class to the noise estimation needs by
    providing the needed Pz attribute, the state uncertainty transformed
    to the measurement space"""
    def update(self, z, R=None, UT=None, hx_args=()):
        if z is None:
            return

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if UT is None:
            UT = unscented_transform

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R

        for i in range(self._num_sigmas):
            self.sigmas_h[i] = self.hx(self.sigmas_f[i], *hx_args)

        # mean and covariance of prediction passed through unscented transform
        zp, Pz = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)

        # compute cross variance of the state and the measurements
        Pxz = zeros((self._dim_x, self._dim_z))
        for i in range(self._num_sigmas):
            dx = self.residual_x(self.sigmas_f[i], self.x)
            dz =  self.residual_z(self.sigmas_h[i], zp)
            Pxz += self.Wc[i] * outer(dx, dz)


        self.K = dot(Pxz, inv(Pz))        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual

        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot3(self.K, Pz, self.K.T)

        self.log_likelihood = logpdf(self.y, np.zeros(len(self.y)), Pz)
        # Store transformed state uncertainty but without added sensor noise
        self.Pz = Pz - R
