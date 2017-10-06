import numpy as np
import json
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from noiseestimation.sensor import LinearSensor
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise_approx,
    estimate_noise_mehra)

# parameters
runs = 200
sample_size = 200
skip_samples = 50
used_taps = int(sample_size * 0.5)
measurement_var = 0.01
filter_misestimation_factor = 1.0


def plot_results(readings, filtered, residuals):
    plt.plot(readings[:, 0, 0], 'go', label="Measurements")
    plt.plot(filtered[:, 0, 0], 'm', linewidth=3, label="Filter")
    plt.legend(loc="lower right")
    plt.xlim([0, 100])
    plt.title("Kalman filtering of position")
    plt.xlabel("Sample")
    plt.ylabel("x (m)")

    plt.show()


def setup():
    # set up sensor simulator
    dt = 0.1
    F = np.array([[1, dt],
                  [0,  1]])
    H = np.array([[1, 0]])
    x0 = np.array([[0],
                   [0.1]])
    sim = LinearSensor(x0, F, H)
    # set up kalman filter
    tracker = KalmanFilter(dim_x=2, dim_z=1)
    tracker.F = F
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)
    tracker.Q = q
    tracker.H = H
    tracker.R = measurement_var * filter_misestimation_factor
    tracker.x = np.array([[0, 0]]).T
    tracker.P = np.eye(2) * 500
    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [[[measurement_var]]] * (sample_size + skip_samples)
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
        residuals.extend(tracker.y[0])

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    return readings, truths, filtered, residuals


def run_tracker():
    sim, tracker = setup()
    readings, truths, filtered, residuals = filtering(sim, tracker)
    # plot_results(readings, filtered, residuals)

    # perform estimation
    cor = Correlator(residuals[-sample_size:])
    correlation = cor.autocorrelation(used_taps)
    R_mehra = estimate_noise_mehra(correlation, tracker.K, tracker.F, tracker.H)
    R_approx = estimate_noise_approx(correlation[0], tracker.H, tracker.P)
    abs_err_approx = R_approx - measurement_var
    rel_err_approx = abs_err_approx / measurement_var
    abs_err_mehra = R_mehra - measurement_var
    rel_err_mehra = abs_err_mehra / measurement_var
    print("True: %.6f" % measurement_var)
    print("Filter: %.6f" % tracker.R)

    print("Estimated (approximation): %.6f" % R_approx)
    print("Absolute error: %.6f" % abs_err_approx)
    print("Relative error: %.6f %%" % (rel_err_approx * 100))

    print("Estimated (mehra): %.6f" % R_mehra)
    print("Absolute error: %.6f" % abs_err_mehra)
    print("Relative error: %.6f %%" % (rel_err_mehra * 100))
    print("-" * 15)
    return (abs_err_mehra, rel_err_mehra, abs_err_approx, rel_err_approx)


if __name__ == "__main__":
    sum = np.zeros(4)
    for i in range(runs):
        print("%d / %d" % (i+1, runs))
        sum += np.abs(run_tracker()).reshape(4,)

    print("Average Mehra: %.6f %%" % (sum * 100 / runs)[1])
    print("Average Mohamed: %.6f %%" % (sum * 100 / runs)[3])

    with open("results.json", "w") as outfile:
        json.dump((sum / runs).tolist(), outfile)
        outfile.close()
