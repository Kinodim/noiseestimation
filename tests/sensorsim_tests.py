import numpy as np
from nose.tools import *
from noiseestimation.sensorsim import *

class TestSensorSim:

    def setup(self):
        self.measurement_std = [1, 2, 3]
        self.sim1d = SensorSim( 0, 0.5, self.measurement_std, 1)
        self.sim2d = SensorSim((0,0), (1, 2), self.measurement_std, 2)

    def basic_constructor_test(self):
        pass

    @raises(ValueError)
    def constructor_measurement_check_test(self):
        sim = SensorSim(0, 1, 4, 1)

    @raises(ValueError)
    def constructor_position_check_test(self):
        sim = SensorSim(0, (1, 1), [4], 2)
    
    @raises(ValueError)
    def constructor_velocity_check_test(self):
        sim = SensorSim((0, 0), 1, [4], 2)

    def read_all_1d_test(self):
        for x in self.measurement_std:
            meas, truth = self.sim1d.read()
            assert meas.shape == (1,1)
            assert truth.shape == (1,1)

    def read_all_2d_test(self):
        for x in self.measurement_std:
            meas, truth = self.sim2d.read()
            assert meas.shape == (2,1)
            assert truth.shape == (2,1)

    @raises(IndexError)
    def read_too_many_test(self):
        for x in self.measurement_std:
            self.sim2d.read()
        self.sim2d.read()

    def batch_read_2d_test(self):
        res = self.sim2d.batch_read()
        assert len(res) == len(self.measurement_std)
        shape = np.asarray(res).shape
        assert shape[0] == len(self.measurement_std)
        assert shape[1] == 2
        assert shape[2] == 2
        assert shape[3] == 1
