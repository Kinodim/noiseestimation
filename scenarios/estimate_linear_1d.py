import numpy as np
from math import fabs
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from noiseestimation.sensor import LinearSensor
from noiseestimation.correlator import Correlator
from noiseestimation.noiseestimator import (
    estimate_noise,
    estimate_noise_approx,
    estimate_noise_mehra)

# parameters
runs = 100
sample_size = 100
used_taps = int(sample_size * 0.5)
measurement_var = 9
filter_misestimation_factor = 5.0


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
    Rs = [[[measurement_var]]] * sample_size
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
    # plot_results(readings, mu, error, residuals)

    # perform estimation
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(used_taps)
    R = estimate_noise(correlation, tracker.K, tracker.F, tracker.H)
    R_mehra = estimate_noise_mehra(correlation, tracker.K, tracker.F, tracker.H)
    R_approx = estimate_noise_approx(correlation[0], tracker.H, tracker.P)
    abs_err = R - measurement_var
    rel_err = abs_err / measurement_var
    print("True: %.3f" % measurement_var)
    print("Filter: %.3f" % tracker.R)
    print("Estimated: %.3f" % R)
    print("Estimated (approximation): %.3f" % R_approx)
    print("Estimated (mehra): %.3f" % R_mehra)
    print("Absolute error: %.3f" % abs_err)
    print("Relative error: %.3f %%" % (rel_err * 100))
    print("-" * 15)
    return rel_err


if __name__ == "__main__":
    sum = .0
    for i in range(runs):
        print("%d / %d" % (i+1, runs))
        sum += fabs(run_tracker())

    print("Avg relative error: %.3f %%" % (sum * 100 / runs))
