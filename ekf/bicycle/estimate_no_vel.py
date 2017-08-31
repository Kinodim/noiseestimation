from __future__ import print_function
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor
from bicycle_ekf import BicycleEKF_noVel
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise_mehra,
    estimate_noise_approx
)

# parameters
skip_samples = 500
used_taps = 150
measurement_var = 1e-4
sim_var = 0.005
num_samples = skip_samples + 300
# num_samples = 11670
dt = 0.01


def setup():
    sim = PlaybackSensor("data/vehicle_state.json",
                         ["fStwAng", "fVx", "fYawrate"])
    # set up kalman filter
    tracker = BicycleEKF_noVel(dt)
    tracker.R = sim_var + measurement_var
    tracker.x = np.array([[0, 0]]).T
    tracker.P = np.eye(2) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [np.eye(1) * sim_var] * num_samples
    readings, filtered, residuals, Ps = [], [], [], []
    for R in Rs:
        time, reading = sim.read(R)
        # skip low velocities
        if reading[1, 0] < 0.03:
            continue
        controls = reading[0:2]
        psi_d = reading[2, 0]
        tracker.predict(controls)
        tracker.update(psi_d)
        readings.append(reading)
        filtered.append(copy(tracker.x))
        Ps.append(copy(tracker.P))
        residuals.append(tracker.y)

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    residuals = np.asarray(residuals)
    Ps = np.asarray(Ps)
    return readings, filtered, residuals, Ps


def perform_estimation(residuals, tracker):
    cor = Correlator(residuals)
    C_arr = cor.autocorrelation(used_taps)
    print("Truth:\n", sim_var)
    R = estimate_noise_mehra(C_arr, tracker.K, tracker.F, tracker.H)
    print("Mehra Method:\n", R)
    R_approx = estimate_noise_approx(C_arr[0], tracker.H, tracker.P)
    print("Approximated Method:\n", R_approx)


def plot_results(readings, filtered, Ps):
    plot_filtered_values(readings, filtered, Ps)
    plot_position(readings, filtered)


def plot_filtered_values(readings, filtered, Ps):
    f, axarr = plt.subplots(4)
    axarr[0].set_title("Schwimmwinkel (deg)")
    axarr[0].plot(
        filtered[:, 0] * 180.0 / np.pi,
        'go')
    axarr[0].set_ylim((-20, 20))
    axarr[1].set_title("Geschaetze Varianz des Schwimmwinkels")
    axarr[1].plot(
        Ps[:, 0, 0]
    )
    axarr[1].set_ylim((0, 0.005))

    axarr[2].set_title("Gierrate (deg/s)")
    axarr[2].plot(
        filtered[:, 1] * 180.0 / np.pi,
        'r-')
    axarr[2].plot(
        readings[:, 2] * 180.0 / np.pi,
        'kx'
    )
    axarr[3].set_title("Geschaetze Varianz der Gierrate")
    axarr[3].plot(
        Ps[:, 1, 1]
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
        velocity = readings[idx, 1, 0]
        delta = dt * velocity * np.array([np.cos(angle),
                                          np.sin(angle)])
        positions[idx + 1] = positions[idx] + delta

    plt.plot(positions[:, 0], positions[:, 1], 'b-')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, filtered, residuals, Ps = filtering(sim, tracker)
    perform_estimation(residuals[skip_samples:], tracker)

    plot_results(readings, filtered, Ps)


if __name__ == "__main__":
    run_tracker()
