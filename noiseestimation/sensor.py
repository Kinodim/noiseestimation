import numpy as np
import numpy.random as rnd


class Sensor:
    """
    TODO DOCU
    """

    def __init__(self, x0, f, h):
        self.x = x0
        self.f = f
        self.h = h

    def step(self, *f_args):
        self.x = self.f(self.x, *f_args)

    def read(self, R, *h_args):
        y = self.h(self.x, *h_args)
        noise = rnd.multivariate_normal(np.zeros(len(R)), R).reshape(-1, 1)
        return y + noise


class LinearSensor(Sensor):
    """
    TODO DOCU
    """

    def __init__(self, x0, F, H):
        def f(x):
            return np.dot(F, x)

        def h(x):
            return np.dot(H, x)
        Sensor.__init__(self, x0, f, h)
