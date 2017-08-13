from __future__ import print_function
import numpy as np
from math import sin, tan, cos
from simple_bicycle_ekf import SimpleBicycleEKF
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
num_samples = 600
used_taps = num_samples / 2
dt = 0.1
measurement_var = 0.01
var_vel = 0.001
var_steer = 0.0001
R_proto = np.array([[1, 0],
                    [0, 1]])
filter_misestimation_factor = 1
turning_threshold_angle = 0.001
wheelbase = 1


# move simulated robot
def f(x, u):
    heading = x[2, 0]
    vel = u[0, 0]
    steering_angle = u[1, 0]
    dist = vel * dt

    # check whether robot is turning
    if abs(steering_angle) > turning_threshold_angle:
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase / tan(steering_angle)
        dx = np.array([[-r * sin(heading) + r * sin(heading + beta)],
                       [r * cos(heading) - r * cos(heading + beta)],
                       [beta]])
    else:
        dx = np.array([[dist * cos(heading)],
                       [dist * sin(heading)],
                       [0]])

    return x + dx


def matrix_error(estimate, truth):
    return np.sqrt(np.sum(np.square(truth - estimate)))


def setup():
    # set up sensor simulator
    H = np.array([[1, 0, 0],
                  [0, 1, 0]])

    def h(x):
        return np.dot(H, x)

    x0 = np.array([[0],
                   [-1],
                   [0]])
    sim = Sensor(x0, f, h)

    # set up kalman filter
    tracker = SimpleBicycleEKF(dt, wheelbase, var_vel, var_steer)
    tracker.R = R_proto * measurement_var * filter_misestimation_factor
    tracker.x = np.array([[0, 0, 0]]).T
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * measurement_var] * num_samples
    readings, truths, filtered, residuals, Ps, Fs = [], [], [], [], [], []
    cmd = np.array([[1],
                    [0.1]])
    for R in Rs:
        sim.step(cmd)
        reading = sim.read(R)
        tracker.predict(cmd)
        tracker.update(reading)
        readings.append(reading)
        truths.append(sim.x)
        filtered.append(tracker.x)
        Ps.append(tracker.P)
        residuals.append(tracker.y)
        Fs.append(tracker.F)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    Ps = np.asarray(Ps)
    residuals = np.asarray(residuals)
    Fs = np.asarray(Fs)
    return readings, truths, filtered, residuals, Ps, Fs


def perform_estimation(residuals, tracker, H, F_arr):
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(used_taps)
    R = estimate_noise(
        correlation, tracker.K, tracker.F, H)
    R_extended = estimate_noise_extended(
        correlation, tracker.K, F_arr, H)
    R_mehra = estimate_noise_mehra(
        correlation, tracker.K, tracker.F, H)
    R_approx = estimate_noise_approx(
        correlation[0], H, tracker.P)
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
    error = matrix_error(R, truth)
    return error


def plot_results(readings, filtered, truths, Ps):
    f, axarr = plt.subplots(2)
    axarr[0].plot(
        readings[:, 0],
        readings[:, 1],
        'go', label="Readings"
    )
    axarr[0].plot(
        truths[:, 0],
        truths[:, 1],
        'k', linewidth=3, label="Truth")
    axarr[0].plot(
        filtered[:, 0],
        filtered[:, 1],
        'm', linewidth=3, label="Filter")
    axarr[0].legend(loc="lower right")
    axarr[0].set_title("Kalman filtering of position")
    # axarr[0].axis('scaled')

    axarr[1].plot(Ps[:, 0, 0], label="X Variance")
    axarr[1].plot(Ps[:, 1, 1], label="Y Variance")
    axarr[1].legend(loc="upper right")

    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, truths, filtered, residuals, Ps, Fs = filtering(sim, tracker)
    perform_estimation(residuals, tracker, tracker.H, Fs[::-1])
    plot_results(readings, filtered, truths, Ps)


if __name__ == "__main__":
    run_tracker()
