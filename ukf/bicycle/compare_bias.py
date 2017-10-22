from __future__ import print_function
import tqdm
import numpy as np
from multiprocessing import Pool
from copy import copy
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor
from bicycle_ukf import BicycleUKFBias
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise_ukf_ml,
    estimate_noise_ukf_map,
    estimate_noise_ukf_scaling
)

# parameters
runs = 100
skip_samples = 300
sample_size = 300
used_taps = sample_size / 2

measurement_var = 1e-6
R_proto = np.array([[1, 0],
                    [0, 2]])
filter_misestimation_factor = 1
map_b = 0.9999
sim_var = 1e-3

dt = 0.01

def matrix_error(estimate, truth):
    return np.sqrt(np.sum(np.square(truth - estimate)))


def setup():
    sim = PlaybackSensor("data/vehicle_state.json",
                         fields=["fYawrate", "fVx"],
                         control_fields=["fStwAng", "fAx"])
    # set up kalman filter
    tracker = BicycleUKFBias(dt)
    Q_factor = np.array(5e-7)
    tracker.Q = np.diag([1, 1, 1, 1e-11] * Q_factor)
    tracker.R = R_proto * (sim_var + measurement_var) * filter_misestimation_factor
    tracker.x = np.array([0, 0, 1, 0.35 * np.pi / 180]).T
    tracker.P = np.eye(4) * 1e0
    tracker.P[3, 3] = 1e-8

    return sim, tracker


def filtering(sim, tracker):
    R = R_proto * sim_var

    readings, filtered, residuals, Ps, map_estimations = [], [], [], [], [0]
    r_mean = 0
    estimation_started = False
    for index in range(sample_size + skip_samples):
        time, reading = sim.read(R)
        measurement = reading[0:2]
        controls = reading[2:]
        # skip low velocities
        if measurement[1, 0] < 0.5:
            continue
        tracker.predict(fx_args=controls)
        tracker.update(measurement[:, 0])

        readings.append(reading)
        filtered.append(copy(tracker.x[:, np.newaxis]))
        Ps.append(copy(tracker.P))
        residual = tracker.y[:, np.newaxis]
        residuals.append(residual)

        if index < skip_samples:
            continue
        if not estimation_started:
            starting_index = index
            estimation_started = True

        average_factor = (1 - map_b) / (1 -
                                        map_b**(index - starting_index + 1))
        r_mean = (1-average_factor) * r_mean + average_factor * residual
        residual -= r_mean
        estimate = estimate_noise_ukf_map(
            residual, tracker.Pz, average_factor,
            map_estimations[-1])
        map_estimations.append(estimate)

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    residuals = np.asarray(residuals)
    Ps = np.asarray(Ps)
    return readings, filtered, residuals, Ps, map_estimations[-1]


def perform_estimation(residuals, tracker, map_estimate):
    residuals = residuals - np.average(residuals, axis=0)
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(used_taps)
    R_ml = estimate_noise_ukf_ml(correlation[0], tracker.Pz)
    R_scaled = estimate_noise_ukf_scaling(correlation[0], tracker.Pz, tracker.R)
    R_map = map_estimate
    truth = R_proto * sim_var
    error_ml = matrix_error(R_ml, truth)
    error_scaled = matrix_error(R_scaled, truth)
    error_map = matrix_error(R_map, truth)

    # truth_norm = matrix_error(truth, 0)
    # print("Truth")
    # print(truth)
    # print("ML:")
    # print("", R_ml)
    # print("\tRelative error: %.6f" % (error_ml / truth_norm))
    # print("Scaled:")
    # print("", R_scaled)
    # print("\tRelative error: %.6f" % (error_scaled / truth_norm))
    # print("MAP:")
    # print("", R_map)
    # print("\tRelative error: %.6f" % (error_map / truth_norm))

    return error_ml, error_scaled, error_map


def plot_results(readings, filtered, Ps, meas_dts):
    plot_filtered_values(readings, filtered, Ps)
    # plot_position(readings, filtered, meas_dts)


def plot_filtered_values(readings, filtered, Ps):
    f, axarr = plt.subplots(3, 2)
    axarr[0, 0].set_title("Schwimmwinkel (deg)")
    axarr[0, 0].plot(
        filtered[:, 0] * 180.0 / np.pi,
        'go')
    axarr[0, 0].set_ylim((-20, 20))
    axarr[0, 1].set_title("Geschaetze Varianz des Schwimmwinkels")
    axarr[0, 1].plot(
        Ps[:, 0, 0]
    )

    axarr[1, 0].set_title("Gierrate (deg/s)")
    axarr[1, 0].plot(
        readings[:, 0] * 180.0 / np.pi,
        'kx'
    )
    axarr[1, 0].plot(
        filtered[:, 1] * 180.0 / np.pi,
        'r-')
    axarr[1, 1].set_title("Geschaetze Varianz der Gierrate")
    axarr[1, 1].plot(
        Ps[:, 1, 1]
    )

    axarr[2, 0].set_title("Geschwindigkeit (m/s)")
    axarr[2, 0].plot(readings[:, 1], 'kx')
    axarr[2, 0].plot(filtered[:, 2], 'b-')
    axarr[2, 1].set_title("Geschaetze Varianz der Geschwindigkeit")
    axarr[2, 1].plot(
        Ps[:, 2, 2]
    )
    plt.show()


def plot_filtered_values_bias(readings, filtered, Ps):
    f, axarr = plt.subplots(4, 2)
    axarr[0, 0].set_title("Schwimmwinkel (deg)")
    axarr[0, 0].plot(
        filtered[:, 0] * 180.0 / np.pi,
        'go')
    axarr[0, 0].set_ylim((-20, 20))
    axarr[0, 1].set_title("Geschaetze Varianz des Schwimmwinkels")
    axarr[0, 1].plot(
        Ps[:, 0, 0]
    )
    axarr[0, 1].set_ylim((0, 0.005))

    axarr[1, 0].set_title("Gierrate (deg/s)")
    axarr[1, 0].plot(
        readings[:, 0] * 180.0 / np.pi,
        'kx'
    )
    axarr[1, 0].plot(
        filtered[:, 1] * 180.0 / np.pi,
        'r-')
    axarr[1, 1].set_title("Geschaetze Varianz der Gierrate")
    axarr[1, 1].plot(
        Ps[:, 1, 1]
    )

    axarr[2, 0].set_title("Gierratenbias (deg/s)")
    axarr[2, 0].plot(
        filtered[:, 3] * 180.0 / np.pi,
        'r-')
    axarr[2, 1].set_title("Geschaetze Varianz des Gierratenbias")
    axarr[2, 1].plot(
        Ps[:, 3, 3]
    )

    axarr[3, 0].set_title("Geschwindigkeit (m/s)")
    axarr[3, 0].plot(readings[:, 1], 'kx')
    axarr[3, 0].plot(filtered[:, 2], 'b-')
    axarr[3, 1].set_title("Geschaetze Varianz der Geschwindigkeit")
    axarr[3, 1].plot(
        Ps[:, 2, 2]
    )
    plt.show()


def plot_position(readings, filtered, dts):
    # skip last value in loops
    yaw_angles = [0] * len(filtered)
    for idx, yawrate in enumerate(filtered[:-1, 1, 0]):
        yaw_angles[idx + 1] = yaw_angles[idx] + dts[idx] * yawrate

    positions = np.zeros((len(filtered), 2))
    for idx in range(len(filtered) - 1):
        sideslip = filtered[idx, 0, 0]
        angle = yaw_angles[idx] + sideslip
        velocity = filtered[idx, 2, 0]
        delta = dts[idx] * velocity * np.array([np.cos(angle),
                                                np.sin(angle)])
        positions[idx + 1] = positions[idx] + delta

    plt.plot(positions[:, 0], positions[:, 1], 'b-')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


def plot_noise_matrices(Rs, Rs_estimated, truth):
    f, axarr = plt.subplots(2)
    axarr[0].set_title("R[0, 0]")
    axarr[0].plot(Rs[:, 0, 0])
    axarr[0].plot([truth[0, 0]] * len(Rs))
    axarr[0].plot(Rs_estimated[:, 0, 0], 'o')

    axarr[1].set_title("R[1, 1]")
    axarr[1].plot(Rs[:, 1, 1])
    axarr[1].plot([truth[1, 1]] * len(Rs))
    axarr[1].plot(Rs_estimated[:, 1, 1], 'o')

    plt.show()


def run_tracker(dummy):
    sim, tracker = setup()

    readings, filtered, residuals, Ps, R_map = filtering(sim, tracker)
    errors = perform_estimation(residuals[skip_samples:], tracker, R_map)
    # plot_filtered_values_bias(readings, filtered, Ps)
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
    R_norm = matrix_error(R_proto * sim_var, 0)
    rel_errors = avg_errors / R_norm
    # ddof = 1 assures an unbiased estimate
    # variances = np.var(errors_arr, axis=0, ddof=1)
    print("-" * 20)
    print("Max Error:")
    print("\t", np.max(errors_arr))
    print("ML estimation:")
    print("\tAverage Error: %.6f" % avg_errors[0])
    print("\tRelative Error: %.6f" % rel_errors[0])
    print("Scaled estimation:")
    print("\tAverage Error: %.6f" % avg_errors[1])
    print("\tRelative Error: %.6f" % rel_errors[1])
    print("MAP estimation:")
    print("\tAverage Error: %.6f" % avg_errors[2])
    print("\tRelative Error: %.6f" % rel_errors[2])
