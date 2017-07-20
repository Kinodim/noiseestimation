from __future__ import print_function
import numpy as np
import math
import copy
from scipy.linalg import block_diag
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from noiseestimation.sensor import Sensor
from noiseestimation.correlator import Correlator
from noiseestimation.noiseestimator import (
    estimate_noise,
    estimate_noise_approx,
    estimate_noise_mehra
)

# parameters
num_samples = 60
used_taps = num_samples / 2
dt = 0.1
measurement_var = 0.35
R_proto = np.array([[1, 0],
                    [0, 0.3]])
filter_misestimation_factor = 5.5


# return range and bearing measurement
def h(x):
    distance = (x[0, 0]**2 + x[2, 0] ** 2) ** 0.5
    bearing = math.atan2(x[2, 0], x[0, 0])

    return np.array([[distance],
                     [bearing]])


def H_jacobian_at(x):
    hyp = x[0, 0]**2 + x[2, 0]**2
    dist = hyp ** 0.5
    return np.array(
        [[x[0, 0] / dist, 0., x[2, 0] / dist],
         [-x[2, 0] / hyp, 0., x[0, 0] / hyp]])


def normalize_angle(x):
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x


def custom_residual(a, b):
    res = np.array(
            [[a[0, 0] - b[0, 0]],
             [normalize_angle(a[1, 0] - b[1, 0])]])
    return res


def matrix_error(estimate, truth):
    return np.sqrt(np.sum(np.square(truth - estimate)))


def setup():
    # set up sensor simulator
    F = np.array([[1, dt, 0],
                  [0,  1, 0],
                  [0,  0, 1]])

    def f(x):
        return np.dot(F, x)

    x0 = np.array([[-3],
                   [0.5],
                   [1]])
    sim = Sensor(x0, f, h)

    # set up kalman filter
    tracker = ExtendedKalmanFilter(dim_x=3, dim_z=2)
    tracker.F = F
    q_x = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    q_y = 0.001 * dt**2
    tracker.Q = block_diag(q_x, q_y)
    tracker.R = R_proto * measurement_var * filter_misestimation_factor
    tracker.x = np.array([[2, -0.1, 4]]).T
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * measurement_var] * num_samples
    readings = []
    truths = []
    filtered = []
    Hs = []
    residuals = []
    for R in Rs:
        sim.step()
        reading = sim.read(R)
        tracker.predict()
        # TODO try to put H calculation after update step
        Hs.append(H_jacobian_at(tracker.x))
        tracker.update(reading, H_jacobian_at, h, residual=custom_residual)
        readings.append(reading)
        truths.append(sim.x)
        filtered.append(tracker.x)
        residuals.append(tracker.y)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    Hs = np.asarray(Hs)
    residuals = np.asarray(residuals)
    return readings, truths, filtered, Hs, residuals


def perform_estimation(residuals, tracker, H):
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(used_taps)
    R = estimate_noise(
        correlation, tracker.K, tracker.F, H)
    R_mehra = estimate_noise_mehra(
        correlation, tracker.K, tracker.F, H)
    R_approx = estimate_noise_approx(
        correlation[0], H, tracker.P, "posterior")
    truth = R_proto * measurement_var
    print("Truth:\n", truth)
    print("Estimation:\n", R)
    print("Error:\n", matrix_error(R, truth))
    print("Mehra estimation:\n", R_mehra)
    print("Error:\n", matrix_error(R_mehra, truth))
    print("Approximated estimation:\n", R_approx)
    print("Error:\n", matrix_error(R_approx, truth))
    print("-" * 15)
    return R


def plot_results(readings, filtered, adjusted_filtered, truths):
    f, axarr = plt.subplots(2)
    axarr[0].plot(
        filtered[:, 0],
        filtered[:, 2],
        'm', linewidth=3, label="Filter (normal)")
    axarr[0].plot(
        adjusted_filtered[:, 0],
        adjusted_filtered[:, 2],
        'g', linewidth=3, label="Filter (adjusted)")
    axarr[0].plot(
        truths[:, 0],
        truths[:, 2],
        'k', linewidth=3, label="Truth")
    axarr[0].legend(loc="upper right")
    axarr[0].set_title("Kalman filtering of position")
    axarr[0].set_ylim((0.8,2))

    axarr[1].plot(
        filtered[:, 1],
        'b', linewidth=3, label="Filter")
    axarr[1].legend(loc="lower right")
    axarr[1].set_title("Kalman filtering of v_x")

    # axarr[2].plot(readings[:, 0, 0], 'go', label="Measurements")
    # axarr[2].set_title("Range measurements")

    # axarr[3].plot(readings[:, 1, 0] * 180 / math.pi, 'go', label="Measurements")
    # axarr[3].set_title("Bearing measurements")

    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, truths, filtered, Hs, residuals = filtering(sim, tracker)

    R = perform_estimation(residuals, tracker, Hs[0])
    adjusted_tracker = copy.copy(tracker)
    adjusted_tracker.R = R
    adjusted_sim = copy.copy(sim)
    adjusted_readings, adjusted_truths, adjusted_filtered, adjusted_Hs, adjusted_residuals = filtering(
        adjusted_sim, adjusted_tracker)
    perform_estimation(adjusted_residuals, adjusted_tracker, adjusted_Hs[0])
    adjusted_readings = np.vstack((readings, adjusted_readings))
    adjusted_filtered = np.vstack((filtered, adjusted_filtered))

    normal_readings, normal_truths, normal_filtered, normal_Hs, normal_residuals = filtering(
        sim, tracker)
    normal_filtered = np.vstack((filtered, normal_filtered))
    normal_truths = np.vstack((truths, normal_truths))

    plot_results(adjusted_readings, normal_filtered, adjusted_filtered, normal_truths)


if __name__ == "__main__":
    run_tracker()
