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
skip_samples = 600
used_taps = 100
measurement_var = 1e-5
R_proto = np.array([[1, 0],
                    [0, 0.1]])
vel_threshold = 0.15
sim_var = 1e-5

sim_time = 1  # in seconds
dt = 0.001

Q = 0.003
var_steer = 0.1 * Q
var_acc = Q * 100


def setup():
    sim = PlaybackSensor("data/vehicle_state.json",
                         fields=["fYawrate", "fVx"],
                         control_fields=["fStwAng", "fAx"])
    # set up kalman filter
    tracker = BicycleEKF(dt)
    tracker.R = R_proto * (sim_var + measurement_var)
    tracker.x = np.array([[0, 0, 1e-3]]).T
    tracker.P = np.eye(3) * 500
    # control not as accurate anymore
    tracker.var_steer = var_steer
    tracker.var_acc = var_acc

    return sim, tracker


def filtering(sim, tracker):
    R = R_proto * sim_var
    # wait for first reading with sensible velocity
    while True:
        init_time, reading = sim.read()
        if reading[1, 0] > vel_threshold:
            break
    init_time *= 1e-9  # nanoseconds to seconds
    time = init_time
    measurement = reading[0:2]
    controls = reading[2:]
    tracker.update(measurement)

    prev_time = init_time
    readings, filtered, residuals, Ps, Fs, Ks, meas_dts = (
        [], [], [], [], [], [], [])
    estimation_times = []
    measurement_times = []
    while time - init_time < sim_time:
        while True:
            next_time, next_meas = sim.read(R)
            if next_meas[1, 0] > vel_threshold:
                break
            print("skipped due to vel")
        next_time *= 1e-9
        F_cumul = np.eye(3)
        while time < next_time:
            tracker.predict(controls)
            F_cumul = np.dot(tracker.F, F_cumul)
            filtered.append(copy(tracker.x))
            Ps.append(copy(tracker.P))
            time += dt
            estimation_times.append(time)
        measurement = next_meas[0:2]
        controls = next_meas[2:]
        tracker.update(measurement)
        meas_dts.append(next_time - prev_time)
        prev_time = next_time

        readings.append(measurement)
        filtered.append(copy(tracker.x))
        Ps.append(copy(tracker.P))
        estimation_times.append(time)
        measurement_times.append(time)
        residuals.append(tracker.y)
        Fs.append(F_cumul)
        Ks.append(tracker.K)

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    residuals = np.asarray(residuals)
    Ps = np.asarray(Ps)
    Fs = np.asarray(Fs)
    Ks = np.asarray(Ks)
    time_offset = estimation_times[0]
    estimation_times = np.asarray(estimation_times) - time_offset
    measurement_times = np.asarray(measurement_times) - time_offset
    meas_dts = meas_dts[1:]
    return (readings, filtered, residuals, Ps, Fs, Ks, meas_dts,
            estimation_times, measurement_times)


def plot_results(readings, filtered, Ps, meas_dts, estimation_times,
                 measurement_times):
    # plot_filtered_values(readings, filtered, Ps,
    #                      estimation_times, measurement_times)
    # plot_position(readings, filtered, meas_dts)
    plot_filtered_velocity(readings, filtered, Ps,
                           estimation_times, measurement_times)


def plot_filtered_values(readings, filtered, Ps, estimation_times,
                         measurement_times):
    f, axarr = plt.subplots(3, 2, sharex=True)
    axarr[0, 0].set_title("Schwimmwinkel")
    axarr[0, 0].set_ylabel(r"$\beta$ (deg)")
    axarr[0, 0].plot(
        estimation_times,
        filtered[:, 0] * 180.0 / np.pi,
        'go', ms=3)
    axarr[0, 0].set_ylim((-20, 20))
    axarr[0, 1].set_title("Zustandsunsicherheiten")
    axarr[0, 1].plot(
        estimation_times,
        Ps[:, 0, 0]
    )
    axarr[0, 1].set_ylabel(r"$\sigma_{\beta}^2$ (deg^2)")

    axarr[1, 0].set_title("Gierrate")
    axarr[1, 0].set_ylabel(r"$\dot{\psi}$ (deg/s)")
    axarr[1, 0].plot(
        measurement_times,
        readings[:, 0] * 180.0 / np.pi,
        'kx'
    )
    axarr[1, 0].plot(
        estimation_times,
        filtered[:, 1] * 180.0 / np.pi,
        'ro', ms=3)
    axarr[1, 1].plot(
        estimation_times,
        Ps[:, 1, 1]
    )
    axarr[1, 1].set_ylabel(r"$\sigma_{\dot{\psi}}^2$ (deg/s)$^2$")

    axarr[2, 0].set_title("Geschwindigkeit")
    axarr[2, 0].set_ylabel(r"$v$ (m/s)")
    axarr[2, 0].plot(
        measurement_times,
        readings[:, 1], 'kx')
    axarr[2, 0].plot(
        estimation_times,
        filtered[:, 2], 'bo', ms=3)
    axarr[2, 0].set_xlabel("t (s)")
    axarr[2, 1].set_title(r"Zustandsunsicherheit Geschwindigkeit ($P_{22}$)")
    axarr[2, 1].plot(
        estimation_times,
        Ps[:, 2, 2]
    )
    axarr[2, 1].set_ylabel(r"$\sigma_v^2$ (m/s)^2")
    axarr[2, 1].set_xlabel("t (s)")

    plt.show()


def plot_filtered_velocity(readings, filtered, Ps, estimation_times,
                         measurement_times):
    f, axarr = plt.subplots(2, 1, sharex=True)
    axarr[0].set_title("Geschwindigkeit")
    axarr[0].set_ylabel(r"$v$ (m/s)")
    axarr[0].plot(
        measurement_times,
        readings[:, 1], 'kx', label="readings")
    axarr[0].plot(
        estimation_times,
        filtered[:, 2], 'bo', ms=3, label="filtered")
    axarr[0].legend(loc="lower right")
    axarr[0].set_xlabel("t (s)")
    axarr[1].set_title(r"Zustandsunsicherheit Geschwindigkeit ($P_{22}$)")
    axarr[1].plot(
        estimation_times,
        Ps[:, 2, 2]
    )
    axarr[1].set_ylabel(r"$\sigma_v^2$ (m/s)$^2$")
    axarr[1].set_xlabel("t (s)")

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


def run_tracker():
    sim, tracker = setup()
    readings, filtered, residuals, Ps, Fs, Ks, meas_dts, estimation_times, measurement_times = filtering(
        sim, tracker)

    plot_results(readings, filtered, Ps, meas_dts, estimation_times, measurement_times)


if __name__ == "__main__":
    run_tracker()
