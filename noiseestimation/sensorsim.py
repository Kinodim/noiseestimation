import numpy as np
from numpy.random import randn

class SensorSim:
    """
    This class provides simulated sensor readings

    Given the initial position and velocity together with an array
    containing the time progression of the measurement noise,
    the read function will output the simulated measurements until
    no more data on its variance is available
    """
    def __init__(self, 
            position,
            velocity,
            measurement_std,
            dim = 1,
            timestep = 0.1):
        self.dim = dim
        self.position = np.asarray(position, dtype="float64").reshape(dim, 1)
        self.velocity = np.asarray(velocity, dtype="float64").reshape(dim, 1)
        if not hasattr(measurement_std, '__iter__'):
            raise ValueError("The measurement noise std argument needs to be iterable")
        self.measurement_std = measurement_std
        self.counter = -1

    def read(self):
        """
        returns both the simulated sensor reading and the actual state as ndarray
        """
        self.counter += 1
        if self.counter >= len(self.measurement_std):
            raise IndexError('No more measurement std data available')

        self.position += self.velocity
        measurement = self.position + self.measurement_std[self.counter] * randn(self.dim, 1)
        return measurement, np.array(self.position) # copy position


    def batch_read(self):
        """
        returns all available data
        """
        batch = np.asarray([ self.read() for _ in self.measurement_std ])
        return batch
