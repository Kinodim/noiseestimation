from __future__ import print_function
import numpy as np
import math
import tqdm
import json
from multiprocessing import Pool
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from noiseestimation.EKF import EstimationExtendedKalmanFilter as EKF
from noiseestimation.sensor import Sensor
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise_approx,
    estimate_noise_mehra,
    estimate_noise_extended
)

# parameters
runs = 400
num_samples = 150
skip_initial = 30
used_taps = num_samples / 2
dt = 0.1
measurement_var = 0.5
R_proto = np.array([[1, 0],
                    [0, 0.2]])
filter_misestimation_factor = 1


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

    x0 = np.array([[-6],
                   [0.5],
                   [1]])
    sim = Sensor(x0, f, h)

    # set up kalman filter
    tracker = EKF(dim_x=3, dim_z=2)
    tracker.F = F
    q_x = Q_discrete_white_noise(dim=2, dt=dt, var=0.0001)
    q_y = 0.0001 * dt**2
    tracker.Q = block_diag(q_x, q_y)
    tracker.R = R_proto * measurement_var * filter_misestimation_factor
    tracker.x = np.array([[-2, -0.1, 2]]).T
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * measurement_var] * num_samples
    (readings, truths, filtered,
     Hs, Ks, residuals, residuals_posterior) = [], [], [], [], [], [], []
    for R in Rs:
        sim.step()
        reading = sim.read(R)
        tracker.predict()
        tracker.update(reading, H_jacobian_at, h, residual=custom_residual)
        # Put H calculation after update step for better estimate
        Hs.append(H_jacobian_at(tracker.x))
        readings.append(reading)
        truths.append(sim.x)
        filtered.append(tracker.x)
        residuals.append(tracker.y)
        residuals_posterior.append(custom_residual(reading, h(tracker.x)))
        Ks.append(tracker.K)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    Hs = np.asarray(Hs)
    Ks = np.asarray(Ks)
    residuals = np.asarray(residuals)
    residuals_posterior = np.asarray(residuals_posterior)
    return readings, truths, filtered, Hs, Ks, residuals, residuals_posterior


def perform_estimation(residuals, residuals_posterior, tracker, H_arr, Ks):
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(used_taps)
    R_mehra = estimate_noise_mehra(
        correlation, Ks[-1], tracker.F, H_arr[-1])
    R_extended = estimate_noise_extended(
        correlation, Ks, tracker.F, H_arr)
    R_approx = estimate_noise_approx(
        correlation[0], H_arr[-1], tracker.P)
    cor_posterior = Correlator(residuals_posterior)
    correlation_posterior = cor_posterior.autocorrelation(used_taps)
    R_approx_posterior = estimate_noise_approx(
        correlation_posterior[0], H_arr[-1], tracker.P, "posterior")

    truth = R_proto * measurement_var
    error_extended = matrix_error(R_extended, truth)
    error_mehra = matrix_error(R_mehra, truth)
    error_approx = matrix_error(R_approx, truth)
    error_approx_posterior = matrix_error(R_approx_posterior, truth)
    return (error_extended, error_mehra,
            error_approx, error_approx_posterior)


def plot_results(readings, filtered, truths):
    f, axarr = plt.subplots(4)
    axarr[0].plot(
        filtered[:, 0],
        filtered[:, 2],
        'm', linewidth=3, label="Filter")
    axarr[0].plot(
        truths[:, 0],
        truths[:, 2],
        'k', linewidth=3, label="Truth")
    axarr[0].legend(loc="lower right")
    axarr[0].set_title("Kalman filtering of position")

    axarr[1].plot(
        filtered[:, 1],
        'b', linewidth=3, label="Filter")
    axarr[1].legend(loc="lower right")
    axarr[1].set_title("Kalman filtering of v_x")

    axarr[2].plot(readings[:, 0, 0], 'go', label="Measurements")
    axarr[2].set_title("Range measurements")

    axarr[3].plot(readings[:, 1, 0] * 180 / math.pi, 'go', label="Measurements")
    axarr[3].set_title("Bearing measurements")

    plt.show()


def run_tracker(dummy):
    sim, tracker = setup()
    (readings, truths,
     filtered, Hs, Ks, residuals, residuals_posterior) = filtering(sim, tracker)
    errors = perform_estimation(
        residuals[skip_initial:], residuals_posterior[skip_initial:],
        tracker, Hs[skip_initial:], Ks[skip_initial:])
    # plot_results(readings, filtered, truths)
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
    # ddof = 1 assures an unbiased estimate
    variances = np.var(errors_arr, axis=0, ddof=1)
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
    print("Approximate estimation (posterior):")
    print("\tAverage Error: %.6f" % avg_errors[3])
    print("\tRelative Error: %.6f" % rel_errors[3])

    res = {}
    res["types"] = ["Extended, Mehra, Mohamed, Mohamed posterior"]
    res["avg_error"] = avg_errors.tolist()
    res["rel_error"] = rel_errors.tolist()
    with open("results.json", "w") as outfile:
        json.dump(res, outfile)
