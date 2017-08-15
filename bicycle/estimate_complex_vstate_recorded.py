import numpy as np
import numpy.random as rnd
from copy import copy
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor
from complex_bicycle_ekf import ComplexBicycleVStateEKF
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise_mehra,
    estimate_noise_approx
)

# parameters
skip_samples = 500
used_taps = 100
measurement_var = 1e-3
R_proto = np.array([[2, 0],
                    [0, 1]])
sim_var = 0.
# num_samples = skip_samples + 500
num_samples = 11670
dt = 0.01


def setup():
    sim = PlaybackSensor("data/vehicle_state.json",
                         ["fStwAng", "fAx", "fYawrate", "fVx"])
    # set up kalman filter
    tracker = ComplexBicycleVStateEKF(dt)
    tracker.R = sim_var + measurement_var
    tracker.x = np.array([[0, 0, 1e-3]]).T
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * sim_var] * num_samples
    readings, filtered, residuals, Ps = [], [], [], []
    for R in Rs:
        time, reading = sim.read([[0]])
        if reading[3, 0] < 0.05:
            continue
        controls = reading[0:2]
        measurement_noise = rnd.multivariate_normal(
            np.zeros(len(R)), R).reshape(-1, 1)
        measurement = reading[2:] + measurement_noise
        tracker.predict(controls)
        tracker.update(measurement)
        readings.append(reading)
        filtered.append(copy(tracker.x))
        Ps.append(copy(tracker.P))
        residuals.append(tracker.y)
        if tracker.K[1,1] > 10:
            print(tracker.K[1,1])
            print(reading[3, 0])
            print(tracker.P)
            print(tracker.F)
            print("-" *15)

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    residuals = np.asarray(residuals)
    Ps = np.asarray(Ps)
    return readings, filtered, residuals, Ps


def perform_estimation(residuals, tracker):
    cor = Correlator(residuals)
    C_arr = cor.autocorrelation(used_taps)
    R = estimate_noise_mehra(C_arr, tracker.K, tracker.F, tracker.H)
    print(R)
    R_approx = estimate_noise_approx(C_arr[0], tracker.H, tracker.P)
    print(R_approx)


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
        filtered[:, 1] * 180.0 / np.pi,
        'ro')
    axarr[1, 0].plot(
        readings[:, 2] * 180.0 / np.pi,
        'k-'
    )
    axarr[1, 1].set_title("Geschaetze Varianz der Gierrate")
    axarr[1, 1].plot(
        Ps[:, 1, 1]
    )

    axarr[2, 0].set_title("Geschwindigkeit (m/s)")
    axarr[2, 0].plot(filtered[:, 2], 'bo')
    axarr[2, 0].plot(readings[:, 3], 'k-')
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


def run_tracker():
    sim, tracker = setup()
    readings, filtered, residuals, Ps = filtering(sim, tracker)
    # perform_estimation(residuals[skip_samples:], tracker)

    plot_results(readings, filtered, Ps)


if __name__ == "__main__":
    run_tracker()
