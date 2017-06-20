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
            measurement_std,
            position = (0,0),
            velocity = (0,0),
            timestep = 0.1):
        self.position = [ position[0], position[1] ] # needs to be mutable
        self.velocity = velocity
        self.measurement_std = measurement_std
        self.counter = -1

    def read(self):
        """
        returns both the simulated sensor reading and the actual state as ndarray
        """
        self.counter += 1
        if self.counter >= len(self.measurement_std):
            raise IndexError('No more measurement std data available')

        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        measurement = np.asarray([ self.position[0] + randn() * self.measurement_std[self.counter],
                 self.position[1] + randn() * self.measurement_std[self.counter] ]).reshape(2,1)
        truth = np.asarray(self.position).reshape(2,1)
        return measurement, truth


    def batch_read(self):
        """
        returns all available data
        """
        batch = np.asarray([ self.read() for _ in self.measurement_std ])
        return batch
