from __future__ import print_function
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor
from bicycle_ukf import BicycleUKF
from noiseestimation.estimation import estimate_noise_ukf_map

# parameters
skip_samples = 200
num_samples = 1200
average_coefficient = 0.3

measurement_var = 1e-5
R_proto = np.array([[1, 0],
                    [0, 3]])
filter_misestimation_factor = 1
sim_var = 1e-3
vel_threshold = 0.3

b = 1 - 1e-2

dt = 0.01


def matrix_error(estimate, truth):
    return np.sqrt(np.sum(np.square(truth - estimate)))


def setup():
    sim = PlaybackSensor("data/vehicle_state.json",
                         fields=["fYawrate", "fVx"],
                         control_fields=["fStwAng", "fAx"])
    # set up kalman filter
    tracker = BicycleUKF(dt)
    Q_factor = np.array(5e-6)
    tracker.Q = np.diag([1, 1, 5] * Q_factor)
    tracker.R = R_proto * (sim_var + measurement_var) * filter_misestimation_factor
    tracker.x = np.array([0, 0, 1]).T
    tracker.P = np.eye(3) * 1e0

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * sim_var] * num_samples
    readings, filtered, Ps, estimated_Rs = [], [], [],\
        [R_proto * sim_var * filter_misestimation_factor]
    R_mean = 0
    estimation_started = False
    starting_index = 0
    for index, R in enumerate(Rs):
        time, reading = sim.read(R)
        measurement = reading[0:2]
        controls = reading[2:]
        # avoid low velocities
        if measurement[1, 0] < 0.5:
            continue
        tracker.predict(fx_args=controls)
        tracker.update(measurement[:, 0])
        readings.append(reading)
        filtered.append(copy(tracker.x))
        Ps.append(copy(tracker.P))
        residual = tracker.y[:, np.newaxis]

        if index < 200:
            continue
        if not estimation_started:
            print("Started at index", index)
            starting_index = index
            estimation_started = True
        R_estimation, R_mean = perform_estimation(residual, tracker,
                                          index - starting_index,
                                          estimated_Rs[-1], R_mean)
        estimated_Rs.append(R_estimation)
        tracker.R = R_estimation

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    Ps = np.asarray(Ps)
    estimated_Rs = np.asarray(estimated_Rs)
    return readings, filtered, Ps, estimated_Rs


def perform_estimation(residual, tracker, index,
                       old_estimate, old_mean):
    average_factor = (1 - b) / (1 - b**(index + 1))
    mean = (1-average_factor) * old_mean + average_factor * residual
    residual -= mean
    estimate = estimate_noise_ukf_map(residual, tracker.Pz, average_factor,
                                      old_estimate, False)
    return estimate, mean


def evaluate_estimation(estimation):
    truth = R_proto * (measurement_var + sim_var)
    error = matrix_error(estimation, truth)
    print("Truth:\n", truth)
    print("Estimation:\n", estimation)
    print("Error: %.6f" % error)
    print("-" * 15)
    return error


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


def plot_noise_matrices(Rs_estimated, truth):
    Rs_estimated = Rs_estimated[30:]
    truth = truth[30:]

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].set_title(r"$R_{00}$")
    axarr[0].plot(Rs_estimated[:, 0, 0], label="estimations")
    axarr[0].plot(truth[:, 0, 0], label="truth")
    axarr[0].set_ylabel(r"$\sigma_{\dot{\psi}}^2$ (deg/s)$^2$")
    axarr[0].legend(loc="lower right")

    axarr[1].set_title(r"$R_{11}$")
    axarr[1].set_ylabel(r"$\sigma_v^2$ (m/s)$^2$")
    axarr[1].plot(Rs_estimated[:, 1, 1])
    axarr[1].plot(truth[:, 1, 1])
    axarr[1].set_xlabel("Sample")

    plt.show()


def plot_noise_matrices_with_data(Rs, truth, filtered, readings):
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 5

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].set_title("R[0, 0]")
    axarr[0, 0].plot(Rs[:, 0, 0])
    axarr[0, 0].plot([truth[0, 0]] * len(Rs))

    axarr[1, 0].set_title("Gierrate (deg/s)")
    axarr[1, 0].plot(readings[:, 0] * 180.0 / np.pi, 'xk')
    axarr[1, 0].plot(filtered[:, 1] * 180.0 / np.pi)

    axarr[0, 1].set_title("R[1, 1]")
    axarr[0, 1].plot(Rs[:, 1, 1])
    axarr[0, 1].plot([truth[1, 1]] * len(Rs))

    axarr[1, 1].set_title("Geschwindigkeit (m/s)")
    axarr[1, 1].plot(readings[:, 1], 'xk')
    axarr[1, 1].plot(filtered[:, 2])

    plt.show()


def run_tracker():
    sim, tracker = setup()

    # actual filtering
    readings, filtered, Ps, R_estimations = filtering(sim, tracker)

    # plot_noise_matrices_with_data(R_estimations[20:], R_proto * sim_var,
    #                               filtered, readings)
    truths = np.asarray([R_proto * sim_var] * len(R_estimations))
    plot_noise_matrices(R_estimations, truths)


if __name__ == "__main__":
    run_tracker()
