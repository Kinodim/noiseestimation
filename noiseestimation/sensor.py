import numpy as np
import numpy.random as rnd


class Sensor:
    """Sensor class using an internal state space representation

    Args:
        x0 (ndarray): initial state
        f (function): transition function
        h (function): measurement function
    """
    def __init__(self, x0, f, h):
        self.x = x0
        self.f = f
        self.h = h

    def step(self, *f_args):
        """Advances state according to transition function

        Args:
            *f_args: Optional arguments to the transition function
        """
        self.x = self.f(self.x, *f_args)

    def read(self, R, *h_args):
        """Outputs readings according to measurement function

        Args:
            R (ndarray, optional): Desired noise covariance matrix
            *h_args: Optional arguments to the measurement function
        Returns:
            ndarray: Measurement corrupted by noise
        """
        y = self.h(self.x, *h_args)
        noise = rnd.multivariate_normal(np.zeros(len(R)), R).reshape(-1, 1)
        return y + noise


class LinearSensor(Sensor):
    """Class for sensors with linear update and measurement

    Args:
        x0 (ndarray): initial state
        F (ndarray): Transition matrix
        H (ndarray): Measurement matrix
    """
    def __init__(self, x0, F, H):
        def f(x):
            return np.dot(F, x)

        def h(x):
            return np.dot(H, x)
        Sensor.__init__(self, x0, f, h)
