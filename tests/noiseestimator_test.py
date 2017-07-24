import numpy as np
from noiseestimation.noiseestimator import (
    estimate_noise,
    estimate_noise_approx,
    estimate_noise_mehra,
    estimate_noise_extended
)


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
        K = np.asarray([[0.32765478],
                        [0.64824051]])
        F = np.array([[1, 0.1],
                      [0,  1]])
        H = np.array([[1, 0]])
        R = estimate_noise(C_arr, K, F, H)
        assert R.shape == (1, 1)
        R, MH_T = estimate_noise(C_arr, K, F, H, True)
        assert R.shape == (1, 1)
        assert MH_T.shape == (2, 1)

    def test_approximate_estimate(self):
        G = [31.371682426081264]
        H = np.array([[1, 0]])
        P = np.array([[1, 0.2],
                      [0, 0.1]])
        R = estimate_noise_approx(G, H, P)
        assert R.shape == (1, 1)

    def test_mehra_estimate(self):
        C_arr = np.asarray([[[31.371682426081264]],
                            [[-6.9977524908455147]],
                            [[0.70876720569959328]],
                            [[-7.1734736799032239]],
                            [[6.8883804618764914]],
                            [[-2.6670500807302022]]])
        K = np.asarray([[0.32765478],
                        [0.64824051]])
        F = np.array([[1, 0.1],
                      [0,  1]])
        H = np.array([[1, 0]])
        R = estimate_noise_mehra(C_arr, K, F, H)
        assert R.shape == (1, 1)

    def test_extended_estimate(self):
        C_arr = np.asarray([[[31.371682426081264]],
                            [[-6.9977524908455147]],
                            [[0.70876720569959328]],
                            [[-7.1734736799032239]],
                            [[6.8883804618764914]],
                            [[-2.6670500807302022]]])
        K = np.asarray([[0.32765478],
                        [0.64824051]])
        F = np.array([[1, 0.1],
                      [0,  1]])
        H_arr = np.array([[[1, 0]],
                          [[0.9, 0]],
                          [[1.1, 0]],
                          [[1, 0]],
                          [[1, 1.1]]])
        R = estimate_noise_extended(C_arr, K, F, H_arr)
        assert R.shape == (1, 1)
