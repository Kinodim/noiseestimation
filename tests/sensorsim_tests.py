from nose.tools import *
from noiseestimation.sensorsim import *

class TestSensorSim:

    def setup(self):
        self.measurement_std = [1, 2, 3]
        self.sim = SensorSim(self.measurement_std)

    def constructor_test(self):
        pass

    def read_all_test(self):
        for x in self.measurement_std:
            self.sim.read()

    @raises(IndexError)
    def read_too_many_test(self):
        for x in self.measurement_std:
            self.sim.read()
        self.sim.read()
