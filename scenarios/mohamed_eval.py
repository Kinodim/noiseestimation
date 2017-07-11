from __future__ import print_function
import numpy as np
from math import fabs
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from sensorsim import SensorSim
from correlator import Correlator
from noiseestimator import estimate_noise_approx


def plot_results(readings, mu, error, residuals):
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(readings, 'go', label="Measurements")
    axarr[0].plot(mu, 'm', linewidth=3, label="Filter")
    axarr[0].legend(loc="lower right")
    axarr[0].set_xlim([0, 100])
    axarr[0].set_title("Kalman filtering of position")

    axarr[1].plot(error, 'r')
    axarr[1].set_title("Estimation error")

    axarr[2].plot(residuals, 'b')
    axarr[2].set_title("Residuals")

    # cor = Correlator(residuals[-50:])
    # print cor.isWhite()
    # cor = Correlator(residuals)
    # print cor.isWhite()

    plt.show()


def run_tracker():
    # parameters
    filter_misestimation_factor = 1.0
    sample_size = 100
    used_taps = int(sample_size * 0.5)
    measurement_std = 3.5

    # set up sensor simulator
    dt = 0.1
    measurement_std_list = np.asarray([measurement_std] * sample_size)
    sim = SensorSim(0, 0.1, measurement_std_list, 1, timestep=dt)

    # set up kalman filter
    tracker = KalmanFilter(dim_x=2, dim_z=1)
    tracker.F = np.array([[1, dt],
                          [0,  1]])
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)
    tracker.Q = q
    tracker.H = np.array([[1, 0]])
    tracker.R = measurement_std**2 * filter_misestimation_factor
    tracker.x = np.array([[0, 0]]).T
    tracker.P = np.eye(2) * 500

    # perform sensor simulation and filtering
    readings = []
    truths = []
    mu = []
    residuals = []
    for _ in measurement_std_list:
        reading, truth = sim.read()
        readings.extend(reading.flatten())
        truths.extend(truth.flatten())
        tracker.predict()
        tracker.update(reading)
        mu.extend(tracker.x[0])
        residual_posterior = reading - np.dot(tracker.H, tracker.x)
        residuals.extend(residual_posterior[0])

    # error = np.asarray(truths) - mu
    # plot_results(readings, mu, error, residuals)

    # perform estimation
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(used_taps)
    R_approx = estimate_noise_approx(
        correlation[0], tracker.H, tracker.P, 'posterior')
    abs_err = measurement_std**2 - R_approx
    rel_err = abs_err / measurement_std**2
    print("True: %.3f" % measurement_std**2)
    print("Filter: %.3f" % tracker.R)
    print("Estimated (approximation): %.3f" % R_approx)
    print("Absolute error: %.3f" % abs_err)
    print("Relative error: %.3f %%" % (rel_err * 100))
    print("-" * 15)
    return rel_err


if __name__ == "__main__":
    sum = .0
    runs = 100
    for i in range(runs):
        print("%d / %d" % (i+1, runs))
        sum += fabs(run_tracker())

    print("Avg relative error: %.3f %%" % (sum * 100 / runs))
