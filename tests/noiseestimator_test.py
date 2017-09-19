import numpy as np
from noiseestimation.estimation import (
    estimate_noise,
    estimate_noise_approx,
    estimate_noise_mehra,
    estimate_noise_extended,
    estimate_noise_ukf_ml,
    estimate_noise_ukf_map,
    estimate_noise_ukf_scaling
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
        C = [[31.371682426081264]]
        H = np.array([[1, 0]])
        P = np.array([[1, 0.2],
                      [0, 0.1]])
        R = estimate_noise_approx(C, H, P)
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

    def test_ukf_estimate_ml(self):
        C = np.asarray([[0.04271917, -0.00366983],
                        [-0.00366983, 0.01893147]])
        P_zz = np.asarray([[1.01043425e-02, 3.41826761e-05],
                           [3.41826761e-05, 1.00712847e-02]])
        R = estimate_noise_ukf_ml(C, P_zz)
        assert R.shape == (2, 2)

    def test_ukf_estimate_map(self):
        residual = np.array([[0.06583802],
                             [0.30857565]])
        P_zz = np.asarray([[1.01043425e-02, 3.41826761e-05],
                           [3.41826761e-05, 1.00712847e-02]])
        R = estimate_noise_ukf_map(residual, P_zz, 0.5, np.ones((2, 2)), False)
        assert R.shape == (2, 2)

        R = estimate_noise_ukf_map(residual, P_zz, 0.5, np.ones((2, 2)), True)
        assert R.shape == (2, 2)

    def test_ukf_estimate_scaling(self):
        C = np.asarray([[0.04271917, -0.00366983],
                        [-0.00366983, 0.01893147]])
        P_zz = np.asarray([[1.01043425e-02, 3.41826761e-05],
                           [3.41826761e-05, 1.00712847e-02]])
        R_filter = np.asarray([[1, 0.1],
                               [0.1, 1]])
        R = estimate_noise_ukf_scaling(C, P_zz, R_filter)
        assert R.shape == (2, 2)
