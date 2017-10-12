from __future__ import print_function
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor
from bicycle_ekf import BicycleEKF

# parameters
used_taps = 100
measurement_var = 1e-5
R_proto = np.array([[1, 0],
                    [0, 2]])
sim_var = 1e-3
num_samples = 3800
# num_samples = 11670
dt = 0.01

Q = 0.01
var_steer = Q * 0.005
var_acc = Q * 2


def setup():
    sim = PlaybackSensor("data/vehicle_state.json",
                         fields=["fYawrate", "fVx"],
                         control_fields=["fStwAng", "fAx"])
    # set up kalman filter
    tracker = BicycleEKF(dt)
    tracker.R = sim_var + measurement_var
    tracker.x = np.array([[0, 0, 1e-3]]).T
    tracker.P = np.eye(3) * 500
    tracker.var_steer = var_steer
    tracker.var_acc = var_acc

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


def plot_results(readings, filtered, residuals, Ps):
    # plot_filtered_values(readings, filtered, Ps)
    plot_residuals(readings, filtered, residuals)
    # plot_position(readings, filtered)


def plot_residuals(readings, filtered, residuals):
    zoom_start = 500
    zoom_end = zoom_start + 300

    f, axarr = plt.subplots(2, 1, sharex=True)
    axarr[0].set_title("Gierrate")
    axarr[0].plot(
        readings[:, 0] * 180.0 / np.pi,
        'kx'
    )
    axarr[0].plot(
        filtered[:, 1, 0] * 180.0 / np.pi,
        'r-')
    axarr[0].set_ylabel(r"$\dot{\psi}$ (deg/s)")
    axarr[0].set_xlim((zoom_start, zoom_end))
    upper_lim = np.max(filtered[zoom_start:zoom_end, 1, 0]) * 1.2 * 180.0 / np.pi
    lower_lim = np.min(filtered[zoom_start:zoom_end, 1, 0]) * 1.2 * 180.0 / np.pi
    axarr[0].set_ylim((lower_lim, upper_lim))

    mean = np.average(residuals[zoom_start:zoom_end, 0, 0])
    axarr[1].set_title("Innovation der Gierrate")
    axarr[1].set_ylabel(r"$\Delta\dot{\psi}$ (deg/s)")
    axarr[1].plot([zoom_start, zoom_end], [0, 0], label="zero")
    axarr[1].plot([zoom_start, zoom_end], [mean, mean], label="mean")
    axarr[1].plot(residuals[:, 0, 0], 'o', ms=3, label="innovation")
    axarr[1].set_xlim((zoom_start, zoom_end))
    upper_lim = np.max(residuals[zoom_start:zoom_end, 0, 0]) * 1.2
    lower_lim = np.min(residuals[zoom_start:zoom_end, 0, 0]) * 1.2
    axarr[1].set_ylim((lower_lim, upper_lim))
    axarr[1].set_xlabel("Sample")
    axarr[1].legend(loc="lower right")

    plt.show()


def plot_filtered_values(readings, filtered, Ps):
    f, axarr = plt.subplots(3, 1, sharex=True)
    axarr[0].set_title("Schwimmwinkel")
    axarr[0].plot(
        filtered[:, 0] * 180.0 / np.pi,
        'C2o')
    axarr[0].set_ylim((-10, 15))
    axarr[0].set_ylabel(r"$\beta$ (deg)")
    # axarr[0, 1].set_title("Geschaetze Varianz des Schwimmwinkels")
    # axarr[0, 1].plot(
    #     Ps[:, 0, 0]
    # )
    # axarr[0, 1].set_ylim((0, 0.005))

    axarr[1].set_title("Gierrate")
    axarr[1].plot(
        readings[:, 0] * 180.0 / np.pi,
        'kx'
    )
    axarr[1].plot(
        filtered[:, 1] * 180.0 / np.pi,
        'r-')
    axarr[1].set_ylabel(r"$\dot{\psi}$ (deg/s)")
    # axarr[1, 1].set_title("Geschaetze Varianz der Gierrate")
    # axarr[1, 1].plot(
    #     Ps[:, 1, 1]
    # )

    axarr[2].set_title("Geschwindigkeit")
    axarr[2].plot(readings[:, 1], 'kx')
    axarr[2].plot(filtered[:, 2], 'b-')
    axarr[2].set_xlabel("Sample")
    axarr[2].set_ylabel(r"$v$ (m/s)")
    # axarr[2, 1].set_title("Geschaetze Varianz der Geschwindigkeit")
    # axarr[2, 1].plot(
    #     Ps[:, 2, 2]
    # )

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

    plt.plot(positions[:, 0], positions[:, 1], 'C2-')
    plt.plot(positions[0, 0], positions[0, 1], 'bx', label="start")
    plt.plot(positions[-1, 0], positions[-1, 1], 'rx', label="end")
    plt.legend(loc="lower right")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, filtered, residuals, Ps, Fs, Ks = filtering(sim, tracker)

    plot_results(readings, filtered, residuals, Ps)


if __name__ == "__main__":
    run_tracker()
