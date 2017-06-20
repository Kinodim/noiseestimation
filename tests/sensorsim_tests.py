import numpy as np
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
            meas, truth = self.sim.read()
            assert meas.shape == (2,1)
            assert truth.shape == (2,1)

    @raises(IndexError)
    def read_too_many_test(self):
        for x in self.measurement_std:
            self.sim.read()
        self.sim.read()

    def batch_read_test(self):
        res = self.sim.batch_read()
        assert len(res) == len(self.measurement_std)
        shape = np.asarray(res).shape
        assert shape[0] == len(self.measurement_std)
        assert shape[1] == 2
        assert shape[2] == 2
        assert shape[3] == 1
