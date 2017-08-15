from __future__ import print_function
import numpy as np
from math import fabs
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from noiseestimation.sensor import LinearSensor
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import estimate_noise


# parameters
runs = 5
sample_size = 100
used_taps = int(sample_size * 0.5)
measurement_var = 3
filter_misestimation_factor = 2.0
R_proto = np.array([[.1, 0],
                    [0, 3]])
Rs = [R_proto * measurement_var] * sample_size


def plot_results(readings, mu, error, residuals):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(readings[:, 0], 'go', label="Measurements")
    axarr[0].plot(mu[:, 0], 'm', linewidth=3, label="Filter")
    axarr[0].legend(loc="lower right")
    axarr[0].set_xlim([0, 100])
    axarr[0].set_title("Kalman filtering of position")

    axarr[1].plot(residuals[:, 0], 'b')
    axarr[1].set_title("Residuals")

    # axarr[1].plot(error, 'r')
    # axarr[1].set_title("Estimation error")

    plt.show()


def setup():
    # set up sensor simulator
    dt = 0.1
    F = np.array([[1, dt],
                  [0,  1]])
    H = np.array([[1, 0],
                  [0, 1]])
    x0 = np.array([[0],
                   [0.1]])
    sim = LinearSensor(x0, F, H)
    # set up kalman filter
    tracker = KalmanFilter(dim_x=2, dim_z=2)
    tracker.F = F
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)
    tracker.Q = q
    tracker.H = H
    tracker.R = np.eye(2) * measurement_var
    tracker.x = np.array([[0, 0]]).T
    tracker.P = np.eye(2) * 500
    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    readings = []
    truths = []
    filtered = []
    residuals = []
    for R in Rs:
        sim.step()
        reading = sim.read(R)
        tracker.predict()
        tracker.update(reading)
        readings.append(reading)
        truths.append(sim.x)
        filtered.append(tracker.x)
        residuals.append(tracker.y)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    residuals = np.asarray(residuals)
    return readings, truths, filtered, residuals


def matrix_error(estimate, truth):
    return np.sqrt(np.sum(np.square(truth - estimate)))


def perform_estimation(residuals, tracker, lags):
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(lags)
    R, MH_T = estimate_noise(
        correlation, tracker.K, tracker.F, tracker.H, True)
    truth = R_proto * measurement_var
    error = matrix_error(R, truth)
    # since H^T = H = eye(2) we have MH^T = M
    print("Estimated state covariance:\n", MH_T)
    print("True R:\n", truth)
    print("Estimated R:\n", R)
    print("Error:\n", matrix_error(R, truth))
    print("-" * 15)
    return error


def run_tracker():
    sim, tracker = setup()
    readings, truths, filtered, residuals = filtering(sim, tracker)
    plot_results(readings, filtered, [], residuals)

    print("Filter state covariance:\n", tracker.P)
    error = perform_estimation(residuals, tracker, used_taps)
    return error


if __name__ == "__main__":
    sum = .0
    for i in range(runs):
        print("%d / %d" % (i+1, runs))
        sum += fabs(run_tracker())

    print("Average error: %.3f" % (sum / runs))
