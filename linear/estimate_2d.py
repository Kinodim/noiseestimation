from __future__ import print_function
import numpy as np
import json
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from noiseestimation.sensor import LinearSensor
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise_approx,
    estimate_noise_mehra
)


# parameters
runs = 1
sample_size = 200
used_taps = 100
skip_samples = 50
measurement_var = 1
R_proto = np.array([[1, 0.1],
                    [0.1, 0.6]])
Rs = [R_proto * measurement_var] * (sample_size + skip_samples)
filter_misestimation_factor = 1.0


def setup():
    dt = 0.1
    # set up sensor sim
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    x0 = np.array([[0],
                   [0.5],
                   [0],
                   [1]])
    sim = LinearSensor(x0, F, H)
    # set up kalman filter
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    tracker.F = F
    q = Q_discrete_white_noise(dim=2, dt=dt, var=1e-3)
    tracker.Q = block_diag(q, q)
    tracker.H = H
    tracker.R = np.diag([measurement_var, measurement_var]) * \
        filter_misestimation_factor
    # tracker.R = np.array([[ 0.95641957,  0.22179505],
    #                       [ 0.19764957,  0.69235086]])
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500
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
    filtered = np.asarray(filtered)
    truths = np.asarray(truths)
    residuals = np.asarray(residuals)
    return readings, truths, filtered, residuals


def plot_results(readings, filtered):
    plt.plot(readings[:, 0, 0],
             readings[:, 1, 0],
             'go', label="Measurements")
    plt.plot(filtered[:, 0, 0],
             filtered[:, 2, 0],
             'm', linewidth=3, label="Filter")
    plt.legend(loc="lower right")
    # plt.xlim([0, 100])
    plt.title("Kalman filtering of position")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.show()


def matrix_error(estimate, truth):
    return np.sqrt(np.sum(np.square(truth - estimate)))


def perform_estimation(residuals, tracker, lags):
    cor = Correlator(residuals[-sample_size:])
    correlation = cor.autocorrelation(lags)
    R_mehra = estimate_noise_mehra(
        correlation, tracker.K, tracker.F, tracker.H)
    R_approx = estimate_noise_approx(
        correlation[0], tracker.H, tracker.P, "posterior")
    truth = R_proto * measurement_var
    error_mehra = matrix_error(R_mehra, truth)
    error_approx = matrix_error(R_approx, truth)
    print("Truth:\n", truth)
    print("Estimation Mehra:\n", R_mehra)
    print("Error Mehra:\n", error_mehra)
    print("Estimation Mohamed:\n", R_approx)
    print("Error Mohamed:\n", error_approx)
    print("-" * 15)
    return (error_mehra, error_approx)


def run_tracker():
    # set up sensor simulator
    sim, tracker = setup()
    readings, truths, filtered, residuals = filtering(sim, tracker)

    plot_results(readings, filtered)

    error = perform_estimation(residuals, tracker, used_taps)
    return error


if __name__ == "__main__":
    sum = np.zeros(2)
    for i in range(runs):
        print("%d / %d" % (i+1, runs))
        sum += np.abs(run_tracker())

    truth = R_proto * measurement_var
    matrix_size = np.sqrt(np.sum(np.square(truth)))
    print("Avg Mehra: %.6f" % (sum[0] / runs))
    print("Avg Mohamed: %.6f" % (sum[1] / runs))

    results = {}
    results["Mehra absolute"] = sum[0] / runs
    results["Mehra relative"] = sum[0] / matrix_size / runs
    results["Mohamed absolute"] = sum[1] / runs
    results["Mohamed relative"] = sum[1] / matrix_size / runs

    with open("results.json", "w") as outfile:
        json.dump(results, outfile)
