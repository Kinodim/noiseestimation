import numpy as np
from noiseestimation.noiseestimator import estimate_noise, estimate_noise_approx


class TestNoiseEstimator:
    """Tests only the error-free execution of the methods

    """

    def test_estimate(self):
        C_arr = np.asarray([[[31.371682426081264]],
                 [[-6.9977524908455147]],
                 [[0.70876720569959328]],
                 [[-7.1734736799032239]],
                 [[6.8883804618764914]],
                 [[-2.6670500807302022]]])
        K = np.asarray([[ 0.32765478],
            [ 0.64824051]])
        F = np.array([[1, 0.1],
            [0,  1]])
        H = np.array([[1, 0]])
        estimate_noise(C_arr, K, F, H)

    def test_approximate_estimate(self):
        G= [31.371682426081264]
        H = np.array([[1, 0]])
        P = np.array([[1, 0.2],
                      [0, 0.1]])
        estimate_noise_approx(G, H, P)
