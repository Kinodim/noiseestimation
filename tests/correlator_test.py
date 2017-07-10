import numpy as np
import pytest
from numpy.testing import assert_array_equal
from noiseestimation.correlator import Correlator


class TestCorrelator:
    def setup(self):
        arr = np.array([[[1]],
                        [[2]],
                        [[1]]])
        self.cor = Correlator(arr)

        arr_2d = [[[1],
                   [2]],
                  [[2],
                   [3]],
                  [[1],
                   [2]]]
        self.cor_2d = Correlator(arr_2d)

    def test_basic_constructor(self):
        pass

    def test_autocorrelation(self):
        C = self.cor.autocorrelation(2)
        assert len(C) == 3
        assert C[0] == (1 + 4 + 1) / 3.
        assert C[1] == (2 + 2) / 3.
        assert C[2] == 1 / 3.

    def test_autocorrelation_2d(self):
        C = self.cor_2d.autocorrelation(2)
        assert C.shape == (3, 2, 2)

        # check symmetry
        assert_array_equal(C[:, 0, 1], C[:, 1, 0])

        # check autocorrelation of first element
        C_upper_left = C[:, 0, 0]
        assert C_upper_left[0] == (1 + 4 + 1) / 3.
        assert C_upper_left[1] == (2 + 2) / 3.
        assert C_upper_left[2] == 1 / 3.

        # check autocorrelation of second element
        C_lower_right = C[:, 1, 1]
        assert C_lower_right[0] == (4 + 9 + 4) / 3.
        assert C_lower_right[1] == (6 + 6) / 3.
        assert C_lower_right[2] == 4 / 3.

    def test_autocorrelation_coefficients(self):
        rho = self.cor.autocorrelation_coefficients(2)
        assert len(rho) == 3
        assert rho[0] == 1
        assert rho[1] == 2. / 3
        assert rho[2] == 1. / 6

    def test_autocorrelation_coefficients_2d(self):
        rho = self.cor_2d.autocorrelation_coefficients(2)
        assert rho.shape == (3, 2, 2)

        # check symmetry
        assert_array_equal(rho[:, 0, 1], rho[:, 1, 0])

        # check correlation coefficients of first element
        rho_upper_left = rho[:, 0, 0]
        assert rho_upper_left[0] == 1
        assert rho_upper_left[1] == 2. / 3
        assert rho_upper_left[2] == 1. / 6

        # Check correlation coefficients of second element
        rho_lower_right = rho[:, 1, 1]
        assert rho_lower_right[0] == 1
        assert rho_lower_right[1] == 4 / (17. / 3)
        assert rho_lower_right[2] == 4 / 3. / (17. / 3)

        # Check correlation coefficients between elements, should not be perfect
        assert rho[0, 0, 1] != 1

    def test_isWhite(self):
        notwhite = Correlator([.1]*100)
        assert not notwhite.isWhite('mehra')
        assert not notwhite.isWhite('ljung-box')

        white = Correlator([0]*50 + [1] + [0]*50)
        assert white.isWhite('mehra')
        assert white.isWhite('ljung-box')

        with pytest.raises(ValueError):
            white.isWhite('novalidmethod')
