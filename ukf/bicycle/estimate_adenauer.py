from __future__ import print_function
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor
from bicycle_ukf import BicycleUKF
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise_ukf_ml,
    estimate_noise_ukf_scaling
)

# parameters
skip_samples = 10
window_size = 150
num_windows = 80
used_taps = window_size / 2

measurement_var = 1e-5
R_proto = np.array([[1, 0],
                    [0, 3]])
filter_misestimation_factor = 2
sim_var = 1e-3
vel_threshold = 0.3

dt = 0.001

b = 0.4


def setup():
    sim = PlaybackSensor("data/vehicle_state_adenauer.json",
                         fields=["fYawrate", "fVx"],
                         control_fields=["fStwAng", "fAx"])
    # set up kalman filter
    tracker = BicycleUKF(dt)
    Q_factor = np.array(1e-7)
    tracker.Q = np.diag([1, 1, 0.5] * Q_factor)
    tracker.R = R_proto * (sim_var) * filter_misestimation_factor
    tracker.x = np.array([0, 0, 1]).T
    tracker.P = np.eye(3) * 1e0

    return sim, tracker


def __get_initial_readings(sim, tracker):
    # wait for first reading with sensible velocity
    while True:
        init_time, reading = sim.read()
        if reading[1, 0] > vel_threshold:
            break
    init_time *= 1e-9  # nanoseconds to seconds
    measurement = reading[0:2]
    controls = reading[2:]
    tracker.update(measurement[:, 0])
    return init_time, controls


def filtering(sim, tracker, num_samples, R=None,
              init_time=None, init_control=None):
    if R is None:
        R = R_proto * sim_var

    if init_time is None or init_control is None:
        init_time, controls = __get_initial_readings(sim, tracker)
    else:
        controls = init_control

    time = init_time
    prev_time = init_time
    readings, filtered, residuals, Ps, meas_dts = [], [], [], [], []
    while len(filtered) < num_samples:
        while True:
            next_time, next_meas = sim.read(R)
            if next_meas[1, 0] > vel_threshold:
                break
        next_time *= 1e-9
        while time < next_time:
            tracker.predict(fx_args=controls)
            time += dt
        measurement = next_meas[0:2]
        controls = next_meas[2:]
        tracker.update(measurement[:, 0])
        meas_dts.append(next_time - prev_time)
        prev_time = next_time

        readings.append(next_meas)
        filtered.append(copy(tracker.x))
        Ps.append(copy(tracker.P))
        residuals.append(tracker.y[:, np.newaxis])

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    residuals = np.asarray(residuals)
    Ps = np.asarray(Ps)
    # meas_dts = meas_dts[1:]
    meas_dts = np.asarray(meas_dts)
    return readings, filtered, residuals, Ps, time, controls, meas_dts


def perform_estimation(residuals, tracker):
    residuals = residuals - np.average(residuals, axis=0)
    cor = Correlator(residuals)
    C_arr = cor.autocorrelation(used_taps)
    R_ml = estimate_noise_ukf_ml(C_arr[0], tracker.Pz)
    # R_scaled = estimate_noise_ukf_scaling(C_arr[0], tracker.Pz, tracker.R)
    return R_ml


def plot_results(readings, filtered, Ps, meas_dts):
    # plot_filtered_values(readings, filtered, Ps)
    plot_position(readings, filtered, meas_dts)


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
    initial_yaw = 100 * np.pi / 180
    yaw_angles = [0] * len(filtered)
    yaw_angles[0] = initial_yaw
    for idx, yawrate in enumerate(filtered[:-1, 1]):
        yaw_angles[idx + 1] = yaw_angles[idx] + dts[idx, 0] * yawrate

    positions = np.zeros((len(filtered), 2))
    for idx in range(len(filtered) - 1):
        sideslip = filtered[idx, 0]
        angle = yaw_angles[idx] + sideslip
        velocity = filtered[idx, 2]
        delta = dts[idx, 0] * velocity * np.array([np.cos(angle),
                                                   np.sin(angle)])
        positions[idx + 1] = positions[idx] + delta

    plt.plot(positions[:, 0], positions[:, 1], 'b-')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


def plot_noise_matrices(Rs, Rs_estimated, truth):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].set_title(r"$R_{00}$")
    axarr[0].plot(Rs[:, 0, 0], label="weighted average")
    axarr[0].plot(truth[:, 0, 0], label="truth")
    axarr[0].plot(Rs_estimated[:, 0, 0], 'o', label="estimations")
    axarr[0].set_ylabel(r"$\sigma_{\dot{\psi}}^2$ (deg/s)$^2$")
    axarr[0].legend(loc="lower right")

    axarr[1].set_title(r"$R_{11}$")
    axarr[1].set_ylabel(r"$\sigma_v^2$ (m/s)$^2$")
    axarr[1].plot(Rs[:, 1, 1])
    axarr[1].plot(truth[:, 1, 1])
    axarr[1].plot(Rs_estimated[:, 1, 1], 'o')
    axarr[1].set_xlabel("Window")

    plt.show()


def plot_noise_matrices_with_data(Rs, Rs_estimated, truth, filtered, readings):
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 5

    f, axarr = plt.subplots(2)
    axarr[0].set_title(r"$R_{00}$")
    axarr[0].plot(Rs[:, 0, 0], label="weighted average")
    axarr[0].plot(truth[:, 0, 0], label="truth")
    axarr[0].plot(Rs_estimated[:, 0, 0], 'o', label="estimations")
    axarr[0].set_ylabel(r"$\sigma_{\dot{\psi}}^2$ (deg/s)$^2$")
    axarr[0].legend(loc="lower right")

    axarr[1].set_title("Gierrate (deg/s)")
    axarr[1].plot(readings[:, 0] * 180.0 / np.pi, 'xk', label="readings")
    axarr[1].plot(filtered[:, 1] * 180.0 / np.pi, label="filtered")
    axarr[1].set_ylabel(r"$\dot{\psi}$ (deg/s)")
    axarr[1].legend(loc="upper right")

    plt.show()


def run_tracker():
    sim, tracker = setup()

    # skip startup results
    readings, filtered, residuals, Ps, time, controls, dts = filtering(
        sim, tracker, skip_samples)

    Rs = [tracker.R]
    Rs_estimated = [tracker.R]
    R_avg = tracker.R
    sim_vars_slope = np.linspace(5e-5, 2e-3, num_windows/4)
    sim_vars_sine = 1e-3 * (1 + 0.5 * np.sin(np.linspace(0, 2*np.pi, num_windows/4)))
    sim_vars_quadratic = 1e-4 * (1 + np.square(np.linspace(0, 4, num_windows/4)))
    sim_vars = [sim_var] * (num_windows / 8)
    sim_vars = np.append(sim_vars, sim_vars_sine)
    sim_vars = np.append(sim_vars, sim_vars_slope)
    sim_vars = np.append(sim_vars, sim_vars_quadratic)
    sim_vars = np.append(sim_vars, [sim_var] * (num_windows/8))
    truths = [R_proto * sim_vars[0]]
    for i in range(num_windows):
        print("Window", i + 1)
        truth = R_proto * sim_vars[i]
        readings, filtered, residuals, Ps, time, controls, dts = filtering(
            sim, tracker, window_size, truth, time, controls)
        dts = dts.reshape((-1, 1))
        R_estimated = perform_estimation(residuals, tracker)
        Rs_estimated.append(R_estimated)

        update_coefficient = (1 - b) / (1 - b**(i + 1))
        R = R_avg * (1-update_coefficient) + \
            update_coefficient * R_estimated
        R_avg = R
        print(R)
        tracker.R = R
        Rs.append(R)
        truths.append(truth)

        if len(Rs) == 2:
            total_readings = readings
            total_filtered = filtered
            total_Ps = Ps
            total_dts = dts
        else:
            total_readings = np.vstack((total_readings, readings))
            total_filtered = np.vstack((total_filtered, filtered))
            total_Ps = np.vstack((total_Ps, Ps))
            total_dts = np.vstack((total_dts, dts))

    Rs = np.asarray(Rs)
    Rs_estimated = np.asarray(Rs_estimated)
    truths = np.asarray(truths)

    # plot_position(total_readings, total_filtered, total_dts)
    plot_noise_matrices_with_data(Rs, Rs_estimated, truths,
                                  total_filtered, total_readings)
    # plot_filtered_values(total_readings, total_filtered, total_Ps)
    print(np.max(total_dts))
    print(np.min(total_dts))
    # plot_noise_matrices(Rs, Rs_estimated, truths)


if __name__ == "__main__":
    run_tracker()
