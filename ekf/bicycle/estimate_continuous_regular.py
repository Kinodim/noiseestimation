from __future__ import print_function
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor
from bicycle_ekf import BicycleEKF
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise_mehra,
    estimate_noise_extended,
    estimate_noise_approx
)

# parameters
skip_samples = 500
window_size = 150
num_windows = 20
used_taps = window_size / 2
average_coefficient = 0.75
measurement_var = 1e-4
R_proto = np.array([[1, 0],
                    [0, 2]])
sim_var = 0.005
misestimation_factor = 2
# num_samples = skip_samples + window_size + num
# num_samples = 11670
dt = 0.01


def setup():
    sim = PlaybackSensor("data/vehicle_state.json",
                         fields=["fYawrate", "fVx"],
                         control_fields=["fStwAng", "fAx"])
    # set up kalman filter
    tracker = BicycleEKF(dt)
    tracker.R = R_proto * (sim_var + measurement_var) * misestimation_factor
    tracker.x = np.array([[0, 0, 1e-1]]).T
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker, num_samples):
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

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    residuals = np.asarray(residuals)
    Ps = np.asarray(Ps)
    Fs = np.asarray(Fs)
    Ks = np.asarray(Ks)
    return readings, filtered, residuals, Ps, Fs, Ks


def perform_estimation(residuals, tracker, F_arr, K_arr):
    cor = Correlator(residuals)
    C_arr = cor.autocorrelation(used_taps)
    R = estimate_noise_mehra(C_arr, tracker.K, tracker.F, tracker.H)
    R_approx = estimate_noise_approx(C_arr[0], tracker.H, tracker.P)
    R_extended = estimate_noise_extended(C_arr, K_arr, F_arr, tracker.H)
    return R_approx


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


def run_tracker():
    sim, tracker = setup()

    # skip startup results
    filtering(sim, tracker, skip_samples)

    Rs = [tracker.R]
    Rs_estimated = [tracker.R]
    R_avg = tracker.R
    for i in range(num_windows):
        readings, filtered, residuals, Ps, Fs, Ks = filtering(
            sim, tracker, window_size)
        R_estimated = perform_estimation(residuals, tracker, Fs, Ks)
        Rs_estimated.append(R_estimated)
        R = R_avg * average_coefficient + \
            (1-average_coefficient) * R_estimated
        R_avg = R
        tracker.R = R
        Rs.append(R)

    Rs = np.asarray(Rs)
    Rs_estimated = np.asarray(Rs_estimated)

    plot_noise_matrices(Rs, Rs_estimated, R_proto * sim_var)

    # plot_results(readings, filtered, Ps)


if __name__ == "__main__":
    run_tracker()
