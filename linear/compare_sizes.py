import numpy as np
import tqdm
from multiprocessing import Pool
from math import fabs
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from noiseestimation.sensor import LinearSensor
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise,
    estimate_noise_approx,
    estimate_noise_mehra)

# parameters
runs = 100
sample_sizes = [80, 100, 120, 150,
                200, 250, 300, 500, 1000]
measurement_var = 9
filter_misestimation_factor = 1.0


def plot_results(errors, variances):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].bar([x for x in range(len(sample_sizes))],
                 errors, 1, color='green', tick_label=sample_sizes)
    axarr[0].set_title("Relative error")

    axarr[1].bar([x for x in range(len(sample_sizes))],
                 errors, 1, color='blue')
    axarr[1].set_title("Variances")

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


def filtering(sim, tracker, sample_size):
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
        # posterior
        # residuals.extend((np.dot(tracker.H, tracker.x) - reading)[0])

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    return readings, truths, filtered, residuals


def perform_estimation(residuals, tracker, sample_size):
    used_taps = int(sample_size / 2)
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(used_taps)
    R = estimate_noise(correlation, tracker.K, tracker.F, tracker.H)
    # R_mehra = estimate_noise_mehra(
    #     correlation, tracker.K, tracker.F, tracker.H)
    # R_approx = estimate_noise_approx(
    #     correlation[0], tracker.H, tracker.P)
    # R_approx_posterior = estimate_noise_approx(
    #     correlation[0], tracker.H, tracker.P, "posterior")
    return R


def run_tracker(sample_size):
    sim, tracker = setup()
    _, _, _, residuals = filtering(sim, tracker, sample_size)
    R = perform_estimation(residuals, tracker, sample_size)
    return R


def check_sample_size(sample_size):
    error_sum = 0
    Rs = []
    pool = Pool(8)
    args = [sample_size] * runs
    pbar = tqdm.tqdm(total=runs)
    for R in pool.imap_unordered(run_tracker, args, chunksize=2):
        pbar.update()
        abs_err = R - measurement_var
        rel_err = abs_err / measurement_var
        error_sum += fabs(rel_err)
        Rs.append(R)
    pbar.close()
    pool.close()
    pool.join()

    avg_rel_err = error_sum / runs
    # ddof = 1 assures an unbiased estimate
    variance = np.var(Rs, ddof=1)
    return avg_rel_err, variance


if __name__ == "__main__":
    errors, variances = [], []
    for ss in sample_sizes:
        print("-"*15)
        print("Checking sample size %d" % ss)
        print("-"*15)
        error, variance = check_sample_size(ss)
        errors.append(error)
        variances.append(variance)

    plot_results(errors, variances)
