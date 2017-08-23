from __future__ import print_function
import numpy as np
import tqdm
from multiprocessing import Pool
from copy import copy
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor
from complex_bicycle_ekf import ComplexBicycleVStateEKF
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise_mehra,
    estimate_noise_extended,
    estimate_noise_approx
)

# parameters
runs = 400
skip_samples = 500
used_taps = 100
measurement_var = 1e-3
R_proto = np.array([[2, 0],
                    [0, 1]])
sim_var = 0.005
num_samples = skip_samples + 300
dt = 0.01


def setup():
    sim = PlaybackSensor("data/vehicle_state.json",
                         fields=["fYawrate", "fVx"],
                         control_fields=["fStwAng", "fAx"])
    # set up kalman filter
    tracker = ComplexBicycleVStateEKF(dt)
    tracker.R = sim_var + measurement_var
    tracker.x = np.array([[0, 0, 1e-3]]).T
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * sim_var] * num_samples
    readings, filtered, residuals, Ps, Fs, Ks = [], [], [], [], [], []
    for R in Rs:
        time, reading = sim.read(R)
        measurement = reading[0:2]
        controls = reading[2:]
        # skip low velocities
        if measurement[1, 0] < 0.05:
            continue
        tracker.predict(controls)
        tracker.update(measurement)
        readings.append(reading)
        filtered.append(copy(tracker.x))
        Ps.append(copy(tracker.P))
        residuals.append(tracker.y)
        Fs.append(tracker.F)
        Ks.append(tracker.K)
        # Debug output for critical Kalman gain
        # if tracker.K[1, 1] > 10:
        #     print(tracker.K[1, 1])
        #     print(reading[3, 0])
        #     print(tracker.P)
        #     print(tracker.F)
        #     print("-" * 15)

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    residuals = np.asarray(residuals)
    Ps = np.asarray(Ps)
    Fs = np.asarray(Fs)
    Ks = np.asarray(Ks)
    return readings, filtered, residuals, Ps, Fs, Ks


def matrix_error(estimate, truth):
    return np.sqrt(np.sum(np.square(truth - estimate)))


def perform_estimation(residuals, tracker, F_arr, Ks):
    cor = Correlator(residuals)
    C_arr = cor.autocorrelation(used_taps)
    truth = R_proto * sim_var
    R = estimate_noise_mehra(C_arr, tracker.K, tracker.F, tracker.H)
    error_mehra = matrix_error(R, truth)
    R_approx = estimate_noise_approx(C_arr[0], tracker.H, tracker.P)
    error_approx = matrix_error(R_approx, truth)
    R_extended = estimate_noise_extended(C_arr, Ks, F_arr, tracker.H)
    error_extended = matrix_error(R_extended, truth)
    return (error_mehra, error_approx, error_extended)


def plot_results(readings, filtered, Ps):
    plot_filtered_values(readings, filtered, Ps)
    plot_position(readings, filtered)


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

    axarr[2, 0].set_title("Geschwindigkeit (m/s)")
    axarr[2, 0].plot(readings[:, 1], 'kx')
    axarr[2, 0].plot(filtered[:, 2], 'b-')
    axarr[2, 1].set_title("Geschaetze Varianz der Geschwindigkeit")
    axarr[2, 1].plot(
        Ps[:, 2, 2]
    )
    plt.show()


def plot_position(readings, filtered):
    # skip last value in loops
    yaw_angles = [0] * len(filtered)
    for idx, yawrate in enumerate(filtered[:-1, 1, 0]):
        yaw_angles[idx + 1] = yaw_angles[idx] + dt * yawrate

    positions = np.zeros((len(filtered), 2))
    for idx in range(len(filtered) - 1):
        sideslip = filtered[idx, 0, 0]
        angle = yaw_angles[idx] + sideslip
        velocity = filtered[idx, 2, 0]
        delta = dt * velocity * np.array([np.cos(angle),
                                          np.sin(angle)])
        positions[idx + 1] = positions[idx] + delta

    plt.plot(positions[:, 0], positions[:, 1], 'b-')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


def run_tracker(dummy):
    sim, tracker = setup()
    readings, filtered, residuals, Ps, Fs, Ks = filtering(sim, tracker)
    errors = perform_estimation(residuals[skip_samples:], tracker,
                                Fs[skip_samples:], Ks[skip_samples:])
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
    # ddof = 1 assures an unbiased estimate
    variances = np.var(errors_arr, axis=0, ddof=1)
    print("-" * 20)
    print("Mehra estimation:")
    print("\tAverage Error: %.6f" % avg_errors[0])
    print("\tError variance: %.6f" % variances[0])
    print("Approximate estimation:")
    print("\tAverage Error: %.6f" % avg_errors[1])
    print("\tError variance: %.6f" % variances[1])
    print("Extended estimation:")
    print("\tAverage Error: %.6f" % avg_errors[2])
    print("\tError variance: %.6f" % variances[2])
