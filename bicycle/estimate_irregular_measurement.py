from __future__ import print_function
import numpy as np
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
skip_samples = 500
used_taps = 100
measurement_var = 1e-5
R_proto = np.array([[1, 0],
                    [0, 0.1]])
vel_threshold = 0.05
sim_var = 1e-3

sim_time = 8  # in seconds
dt = 0.0001


def setup():
    sim = PlaybackSensor("data/vehicle_state.json",
                         fields=["fYawrate", "fVx"],
                         control_fields=["fStwAng", "fAx"])
    # set up kalman filter
    tracker = ComplexBicycleVStateEKF(dt)
    tracker.R = R_proto * (sim_var + measurement_var)
    tracker.x = np.array([[0, 0, 1e-3]]).T
    tracker.P = np.eye(3) * 500
    # control not as accurate anymore
    tracker.var_steer = 0.0005
    tracker.var_acc = 0.5

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
    while time - init_time < sim_time:
        while True:
            next_time, next_meas = sim.read(R)
            if reading[1, 0] > vel_threshold:
                break
        next_time *= 1e-9
        while time < next_time:
            tracker.predict(controls)
            time += dt
            # filtered.append(copy(tracker.x))
            # Ps.append(copy(tracker.P))
        measurement = next_meas[0:2]
        controls = next_meas[2:]
        tracker.update(measurement)
        meas_dts.append(next_time - prev_time)
        prev_time = next_time

        readings.append(next_meas)
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
    meas_dts = meas_dts[1:]
    return readings, filtered, residuals, Ps, Fs, Ks, meas_dts


def perform_estimation(residuals, tracker, F_arr, K_arr):
    cor = Correlator(residuals)
    C_arr = cor.autocorrelation(used_taps)
    print("Simulated:\n", R_proto * sim_var)
    R = estimate_noise_mehra(C_arr, tracker.K, tracker.F, tracker.H)
    print("Mehra:\n", R)
    R_approx = estimate_noise_approx(C_arr[0], tracker.H, tracker.P)
    print("Approximation:\n", R_approx)
    R_extended = estimate_noise_extended(C_arr, K_arr, F_arr, tracker.H)
    print("Extended:\n", R_extended)


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
    readings, filtered, residuals, Ps, Fs, Ks, meas_dts = filtering(
        sim, tracker)
    perform_estimation(residuals[skip_samples:], tracker,
                       Fs[skip_samples:], Ks[skip_samples:])

    plot_results(readings, filtered, Ps, meas_dts)


if __name__ == "__main__":
    run_tracker()
