import numpy as np
import math
from scipy.linalg import block_diag
from numpy.testing import assert_array_equal
import noiseestimation.sensor as sn


class TestSensor:
    def setup(self):
        x0 = np.array([[1],
                       [2]])

        def f(x, offset):
            return x + offset

        def h(x, x_offset):
            return np.array([[math.sin(x[0, 0]) + x_offset],
                             [math.cos(x[1, 0])]])

        self.testee_2d = sn.Sensor(x0, f, h)
        self.R_2d = np.eye(2) * 1.0

    def test_2d_step(self):
        u = np.array([[1],
                      [2]])
        self.testee_2d.step(u)
        assert_array_equal(
            np.array([[2.0],
                      [4.0]]),
            self.testee_2d.x
        )

    def test_2d_read(self):
        reading = self.testee_2d.read(self.R_2d, 0)
        assert reading.shape == (2, 1)
        x_offset = 0.1
        reading_pure = self.testee_2d.read(np.zeros((2, 2)), x_offset)
        assert_array_equal(
            np.array([[math.sin(1.0) + x_offset],
                      [math.cos(2.0)]]),
            reading_pure
        )


class TestLinearSensor:
    def setup(self):
        self.dt = 0.1
        x0_1d = np.array([[0],
                          [1]])
        x0_2d = np.array([[5],
                          [1],
                          [5],
                          [2]])

        F_1d = np.array([[1, self.dt],
                        [0, 1]])
        F_2d = block_diag(F_1d, F_1d)
        H_1d = np.array([[1, 0]])
        H_2d = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]])

        self.testee_1d = sn.LinearSensor(x0_1d, F_1d, H_1d)
        self.testee_2d = sn.LinearSensor(x0_2d, F_2d, H_2d)

        self.R_1d = np.eye(1) * 1.0
        self.R_2d = np.eye(2) * 1.0

    def test_1d_step(self):
        self.testee_1d.step()
        assert_array_equal(
            np.array([[0.1],
                      [1]]),
            self.testee_1d.x
        )

    def test_1d_read(self):
        reading = self.testee_1d.read(self.R_1d)
        assert reading.shape == (1, 1)
        # 6 sigma bound should not be passed too often during testing
        assert -6.0 * self.R_1d[0, 0] < reading[0, 0] < 6.0 * self.R_1d[0, 0]

    def test_2d_step(self):
        self.testee_2d.step()
        assert_array_equal(
            np.array([[5.1],
                      [1],
                      [5.2],
                      [2]]),
            self.testee_2d.x
        )

    def test_2d_read(self):
        reading = self.testee_2d.read(self.R_2d)
        assert reading.shape == (2, 1)
