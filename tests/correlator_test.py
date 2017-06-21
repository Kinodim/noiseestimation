import numpy as np
from noiseestimation.correlator import *

class TestCorrelator:
    def setup(self):
        arr = [1, 2, 1]
        self.cor = Correlator(arr, 2)

    def test_basic_constructor(self):
        pass

    def test_covariance(self):
        C = self.cor.covariance()
        assert len(C) == 3
        assert C[0] == (1 + 4 + 1) / 3.
        assert C[1] == (2 + 2) / 3.
        assert C[2] == 1 / 3.
        
        # arr = [1, 1, 2, 2, 3, 3, -1, -2, -5, 10]
        # self.cor = Correlator(arr, 5)
