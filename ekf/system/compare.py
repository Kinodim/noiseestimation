from __future__ import print_function
import numpy as np
import tqdm
import json
import copy
from multiprocessing import Pool
from math import sin, tan, cos
from simple_bicycle_ekf import SimpleBicycleEKF
from matplotlib import pyplot as plt
from noiseestimation.sensor import Sensor
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise_approx,
    estimate_noise_mehra,
    estimate_noise_extended
)

# parameters
runs = 100
skip_initial = 50
num_samples = 250
used_taps = (num_samples - skip_initial) / 2
dt = 0.1
Q = 0.01
var_vel = Q
var_steer = Q
measurement_var = 0.05
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
    R = R_proto * measurement_var
    readings, truths, filtered, residuals, Ps, Fs, Ks = [], [], [], [], [], [], []
    cmds = [np.array([[0.6],
                      [0.23]])] * (num_samples / 4)
    cmds.extend([np.array([[2],
                           [-0.20]])] * (num_samples / 2))
    cmds.extend([np.array([[1],
                           [0.2]])] * (num_samples / 4))
    for cmd in cmds:
        sim.step(cmd)
        reading = sim.read(R)
        tracker.predict(cmd)
        tracker.update(reading)
        readings.append(reading)
        truths.append(sim.x)
        filtered.append(copy.copy(tracker.x))
        Ps.append(tracker.P)
        residuals.append(tracker.y)
        Fs.append(tracker.F)
        Ks.append(tracker.K)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    Ps = np.asarray(Ps)
    residuals = np.asarray(residuals)
    Fs = np.asarray(Fs)
    Ks = np.asarray(Ks)
    return readings, truths, filtered, residuals, Ps, Fs, Ks


def perform_estimation(residuals, tracker, H, F_arr, K_arr):
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(used_taps)
    R_extended = estimate_noise_extended(
        correlation, K_arr, F_arr, H)
    R_mehra = estimate_noise_mehra(
        correlation, K_arr[-1], F_arr[-1], H)
    R_approx = estimate_noise_approx(
        correlation[0], H, tracker.P)
    truth = R_proto * measurement_var
    error_extended = matrix_error(R_extended, truth)
    error_mehra = matrix_error(R_mehra, truth)
    error_approx = matrix_error(R_approx, truth)
    return (error_extended, error_mehra, error_approx)


def plot_results(readings, filtered, truths, Ps):
    axarr = [plt.subplot()]

    axarr[0].plot(
        readings[:, 0],
        readings[:, 1],
        'o', label="Readings"
    )
    axarr[0].plot(
        truths[:, 0],
        truths[:, 1],
        'k', linewidth=3, label="Truth")
    axarr[0].plot(
        filtered[:, 0],
        filtered[:, 1],
        linewidth=3, label="Filter")
    axarr[0].legend(loc="lower right")
    axarr[0].set_title("Kalman filtering of position")
    axarr[0].set_xlabel("x (m)")
    axarr[0].set_ylabel("y (m)")
    axarr[0].axis('scaled')

    # axarr[0].plot(Ps[:, 0, 0], 'g', label="X state variance")
    # axarr[0].plot(Ps[:, 1, 1], 'r', label="Y state variance")
    # axarr[0].legend(loc="upper right")
    # axarr[0].set_ylim((0, 0.003))
    # axarr[0].set_ylabel("$\sigma^2$ ($m^2$)")
    # axarr[0].set_xlabel("Sample")
    # axarr[0].set_title("State covariance")

    plt.show()


def run_tracker(dummy):
    sim, tracker = setup()
    readings, truths, filtered, residuals, Ps, Fs, Ks = filtering(sim, tracker)
    errors = perform_estimation(
        residuals[skip_initial:], tracker, tracker.H,
        Fs[skip_initial:], Ks[skip_initial:])
    # plot_results(readings, filtered, truths, Ps)
    state_error = np.sqrt(np.sum(np.square(truths-filtered)[50:, [0, 1], 0]))
    if state_error > 1.5:
        print("WARNING: Outlier (%.6f)" % state_error)
    return errors


if __name__ == "__main__":
    pool = Pool(8)
    args = [0] * runs
    pbar = tqdm.tqdm(total=runs)
    errors_arr = []
    for errors in pool.imap_unordered(run_tracker, args, chunksize=2):
        pbar.update()
        errors_arr.append(errors)
    pbar.close()
    pool.close()
    pool.join()

    errors_arr = np.asarray(errors_arr)
    avg_errors = np.sum(errors_arr, axis=0) / float(runs)
    matrix_size = matrix_error(R_proto * measurement_var, 0)
    rel_errors = avg_errors / matrix_size
    print("Max. error: %.6f" % np.max(errors_arr))
    # ddof = 1 assures an unbiased estimate
    # variances = np.var(errors_arr, axis=0, ddof=1)
    print("-" * 20)
    print("Extended estimation:")
    print("\tAverage Error: %.6f" % avg_errors[0])
    print("\tRelative Error: %.6f" % rel_errors[0])
    print("Mehra estimation:")
    print("\tAverage Error: %.6f" % avg_errors[1])
    print("\tRelative Error: %.6f" % rel_errors[1])
    print("Approximate estimation:")
    print("\tAverage Error: %.6f" % avg_errors[2])
    print("\tRelative Error: %.6f" % rel_errors[2])

    res = {}
    res["types"] = ["Extended", "Mehra", "Mohamed"]
    res["abs_err"] = avg_errors.tolist()
    res["rel_err"] = rel_errors.tolist()
    with open("results.txt", "w") as outfile:
        json.dump(res, outfile)
