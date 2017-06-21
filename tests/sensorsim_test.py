import pytest
import numpy as np
from noiseestimation.sensorsim import *

class TestSensorSim:

    def setup(self):
        self.measurement_std = [1, 2, 3]
        self.sim1d = SensorSim( 0, 0.5, self.measurement_std, 1)
        self.sim2d = SensorSim((0,0), (1, 2), self.measurement_std, 2)

    def test_basic_constructor(self):
        pass

    def test_constructor_measurement_check(self):
        with pytest.raises(ValueError):
            sim = SensorSim(0, 1, 4, 1)

    def test_constructor_position_check(self):
        with pytest.raises(ValueError):
            sim = SensorSim(0, (1, 1), [4], 2)

    def test_constructor_velocity_check(self):
        with pytest.raises(ValueError):
            sim = SensorSim((0, 0), 1, [4], 2)

    def test_read_all_1d(self):
        for x in self.measurement_std:
            meas, truth = self.sim1d.read()
            assert meas.shape == (1,1)
            assert truth.shape == (1,1)

    def test_read_all_2d(self):
        for x in self.measurement_std:
            meas, truth = self.sim2d.read()
            assert meas.shape == (2,1)
            assert truth.shape == (2,1)

    def test_read_too_many(self):
        for x in self.measurement_std:
            self.sim2d.read()
        with pytest.raises(IndexError):
            self.sim2d.read()

    def test_batch_read_2d(self):
        res = self.sim2d.batch_read()
        assert len(res) == len(self.measurement_std)
        shape = np.asarray(res).shape
        assert shape[0] == len(self.measurement_std)
        assert shape[1] == 2
        assert shape[2] == 2
        assert shape[3] == 1
