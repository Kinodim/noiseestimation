from __future__ import print_function
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor
from bicycle_ukf import BicycleUKF
from noiseestimation.estimation import estimate_noise_ukf_map

# parameters
num_samples = 600
used_taps = 50
dt = 0.01
measurement_var = 1e-5
sim_var = 1e-3
R_proto = np.array([[1, 0],
                    [0, 2]])
filter_misestimation_factor = 1


def matrix_error(estimate, truth):
    return np.sqrt(np.sum(np.square(truth - estimate)))


def setup():
    sim = PlaybackSensor("data/vehicle_state.json",
                         fields=["fYawrate", "fVx"],
                         control_fields=["fStwAng", "fAx"])
    # set up kalman filter
    tracker = BicycleUKF(dt)
    Q_factor = np.array(1e-5)
    tracker.Q = np.diag([1, 1, 0.1] * Q_factor)
    tracker.R = np.eye(2) * (sim_var + measurement_var) * 1
    tracker.x = np.array([0, 0, 1]).T
    tracker.P = np.eye(3) * 1e0

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * sim_var] * num_samples
    readings, filtered, Ps, estimated_Rs = [], [], [], [0]
    estimation_started = False
    starting_index = 0
    for index, R in enumerate(Rs):
        time, reading = sim.read(R)
        measurement = reading[0:2]
        controls = reading[2:]
        # skip low velocities
        if measurement[1, 0] < 0.5:
            continue
        tracker.predict(fx_args=controls)
        tracker.update(measurement[:, 0])
        readings.append(reading)
        filtered.append(copy(tracker.x))
        Ps.append(copy(tracker.P))
        residual = tracker.y[:, np.newaxis]
        if not estimation_started:
            print("Started at index", index)
            starting_index = index
            estimation_started = True
        R_estimation = perform_estimation(residual, tracker,
                                          index - starting_index,
                                          estimated_Rs[-1])
        estimated_Rs.append(R_estimation)

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    Ps = np.asarray(Ps)
    return readings, filtered, Ps, estimated_Rs[-1]


def perform_estimation(residual, tracker, index, old_estimate):
    average_factor = (1 - 0.95) / (1 - 0.95**(index + 1))
    estimate = estimate_noise_ukf_map(residual, tracker.Pz, average_factor,
                                      old_estimate, False)
    return estimate


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


def run_tracker():
    sim, tracker = setup()
    readings, filtered, Ps, estimation = filtering(sim, tracker)
    evaluate_estimation(estimation)
    plot_filtered_values(readings, filtered, Ps)


if __name__ == "__main__":
    run_tracker()
