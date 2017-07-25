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
    estimate_noise_mehra,
    estimate_noise_extended
)

# parameters
num_samples = 180
used_taps = 70
dt = 0.1
measurement_var = 0.05
R_proto = np.array([[1, 0],
                    [0, 0.2]])
filter_misestimation_factor = 0.05
Q_var = 0.001


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

    x0 = np.array([[-2],
                   [0.2],
                   [1]])
    sim = Sensor(x0, f, h)

    # set up kalman filter
    tracker = ExtendedKalmanFilter(dim_x=3, dim_z=2)
    tracker.F = F
    q_x = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)
    q_y = Q_var * dt**2
    tracker.Q = block_diag(q_x, q_y)
    tracker.R = R_proto * measurement_var * filter_misestimation_factor
    tracker.x = np.array([[-2, -1, 5]]).T
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * measurement_var] * num_samples
    readings, truths, filtered, Hs, residuals, Ps = [], [], [], [], [], []
    for R in Rs:
        sim.step()
        reading = sim.read(R)
        tracker.predict()
        tracker.update(reading, H_jacobian_at, h, residual=custom_residual)
        # Put H calculation after update step
        Hs.append(H_jacobian_at(tracker.x))
        readings.append(reading)
        truths.append(sim.x)
        filtered.append(tracker.x)
        residuals.append(tracker.y)
        Ps.append(tracker.P)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    Hs = np.asarray(Hs)
    residuals = np.asarray(residuals)
    Ps = np.asarray(Ps)
    return readings, truths, filtered, Hs, residuals, Ps


def perform_estimation(residuals, tracker, Hs):
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(used_taps)
    R = estimate_noise(
        correlation, tracker.K, tracker.F, Hs[0])
    R_extended = estimate_noise_extended(
        correlation, tracker.K, tracker.F, Hs)
    R_mehra = estimate_noise_mehra(
        correlation, tracker.K, tracker.F, Hs[0])
    R_approx = estimate_noise_approx(
        correlation[0], Hs[0], tracker.P, "posterior")
    truth = R_proto * measurement_var
    print("Truth:\n", truth)
    print("Estimation:\n", R)
    print("Error:\n", matrix_error(R, truth))
    print("Extended Estimation:\n", R_extended)
    print("Error:\n", matrix_error(R_extended, truth))
    print("Mehra estimation:\n", R_mehra)
    print("Error:\n", matrix_error(R_mehra, truth))
    print("Approximated estimation:\n", R_approx)
    print("Error:\n", matrix_error(R_approx, truth))
    print("-" * 15)
    return R_extended


def plot_results(readings, filtered, adjusted_filtered, truths, Ps):
    f, axarr = plt.subplots(4)
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
    axarr[0].set_ylim((0.8, 1.2))

    confidences = np.sum(np.diagonal(Ps, axis1=1, axis2=2), axis=1)
    axarr[1].plot(
        confidences,
        'b', linewidth=3)
    axarr[1].set_title("State estimate covariance (P)")
    axarr[1].set_ylim((0, 0.1))

    axarr[2].plot(
        readings[:, 0, 0], 'go', label="Measurements")
    axarr[2].set_title("Range measurements")

    axarr[3].plot(
       readings[:, 1, 0] * 180 / math.pi, 'go', label="Measurements")
    axarr[3].set_title("Bearing measurements")

    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, truths, filtered, Hs, residuals, Ps = filtering(sim, tracker)

    R = perform_estimation(residuals[40:], tracker, Hs[::-1])
    adjusted_tracker = copy.copy(tracker)
    adjusted_tracker.R = R
    adjusted_sim = copy.copy(sim)
    (
        adjusted_readings,
        adjusted_truths,
        adjusted_filtered,
        adjusted_Hs,
        adjusted_residuals,
        adjusted_Ps
    ) = filtering(adjusted_sim, adjusted_tracker)
    # perform_estimation(adjusted_residuals, adjusted_tracker, adjusted_Hs[::-1])
    adjusted_readings = np.vstack((readings, adjusted_readings))
    adjusted_filtered = np.vstack((filtered, adjusted_filtered))
    adjusted_Ps = np.vstack((Ps, adjusted_Ps))

    (
        normal_readings,
        normal_truths,
        normal_filtered,
        normal_Hs,
        normal_residuals,
        normal_Ps
    ) = filtering(sim, tracker)
    normal_filtered = np.vstack((filtered, normal_filtered))
    normal_truths = np.vstack((truths, normal_truths))

    plot_results(
        adjusted_readings,
        normal_filtered,
        adjusted_filtered,
        normal_truths,
        adjusted_Ps)


if __name__ == "__main__":
    run_tracker()
