import numpy as np
import pytest
from noiseestimation.correlator import *

class TestCorrelator:
    def setup(self):
        arr = [1, 2, 1]
        self.cor = Correlator(arr)

    def test_basic_constructor(self):
        pass

    def test_covariance(self):
        C = self.cor.covariance(2)
        assert len(C) == 3
        assert C[0] == (1 + 4 + 1) / 3.
        assert C[1] == (2 + 2) / 3.
        assert C[2] == 1 / 3.

    def test_autocorrelation(self):
        rho = self.cor.autocorrelation(2)
        assert len(rho) == 3
        assert rho[0] == 1
        assert rho[1] == 2. / 3
        assert rho[2] == 1. / 6

    def test_isWhite(self):
        notwhite = Correlator([.1]*100)
        assert notwhite.isWhite('mehra') == False
        assert notwhite.isWhite('ljung-box') == False

        white = Correlator([0]*50 + [1] + [0]*50)
        assert white.isWhite('mehra') == True
        assert white.isWhite('ljung-box') == True

        with pytest.raises(ValueError):
            white.isWhite('novalidmethod')
