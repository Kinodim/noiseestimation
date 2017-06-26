import numpy as np
import numpy.random as rnd

class SensorSim:
    """This class provides simulated sensor readings

    Given the initial position and velocity of an object together with
    an array containing the time progression of the measurement noise,
    this class will simulate the sensor readings for this objects
    position.

    Args:
        position (tuple): Initial object position
        velocity (tuple): Object velocity
        measurement_std (list): Measurement standard deviation for
            all desired timesteps
        dim (int): Filter dimension
        timestep: The discrete timestep between two updates

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
        """Returns a single sensor reading

        Returns the current sensor reading corrupted by noise with the
        standard deviation specified in measurement_std as well as the
        true position. This will lead to an error if the end of this
        array is reached.

        Returns:
            tuple: 2-element tuple containing:
                - ndarray: Simulated sensor reading
                - ndarray: True object position

        Raises:
            IndexError: The end of the measurement_std list has been reached
        
        """
        self.counter += 1
        if self.counter >= len(self.measurement_std):
            raise IndexError('No more measurement std data available')

        self.position += self.velocity
        measurement = self.position + self.measurement_std[self.counter] * rnd.randn(self.dim, 1)
        return measurement, np.array(self.position) # copy position


    def batch_read(self):
        """Returns all the available data

        Returns all the sensor readings and true positions for all the
        entries in the measurement_std list.

        Returns:
            tuple: 2-element tuple containing:
                - ndarray: Simulated sensor readings
                - ndarray: True object positions
        """
        batch = np.asarray([ self.read() for _ in self.measurement_std ])
        return batch
