import numpy as np
from noiseestimation.noiseestimator import estimate_noise


class TestNoiseEstimator:

    def test_estimate(self):
        C_arr = [31.371682426081264, -6.9977524908455147, 0.70876720569959328, -7.1734736799032239, 6.8883804618764914, -2.6670500807302022]
        K = np.asarray([[ 0.32765478],
            [ 0.64824051]])
        F = np.array([[1, 0.1],
            [0,  1]])
        H = np.array([[1, 0]])
        estimate_noise(C_arr, K, F, H)
