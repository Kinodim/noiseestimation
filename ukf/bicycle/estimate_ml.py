from __future__ import print_function
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor
from bicycle_ukf import BicycleUKF
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import estimate_noise_ukf_ml

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
    tracker.R = R_proto * (sim_var + measurement_var) * \
        filter_misestimation_factor
    tracker.x = np.array([0, 0, 0.1]).T
    tracker.P = np.eye(3) * 1e0

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * sim_var] * num_samples
    readings, filtered, residuals, Ps = [], [], [], []
    for idx, R in enumerate(Rs):
        time, reading = sim.read(R)
        measurement = reading[0:2]
        controls = reading[2:]
        # skip low velocities
        if measurement[1, 0] < 0.5:
            continue
        tracker.predict(fx_args=controls)
        tracker.update(measurement[:, 0])

        readings.append(reading)
        filtered.append(copy(tracker.x[:, np.newaxis]))
        Ps.append(copy(tracker.P))
        # debug output if P not positive semidefinite
        # if not np.all(np.linalg.eigvals(tracker.P) >= 0):
        #     print(tracker.Pz)
        #     print(tracker.K)
        #     print(tracker.P)
        residuals.append(tracker.y[:, np.newaxis])

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    residuals = np.asarray(residuals)
    Ps = np.asarray(Ps)
    return readings, filtered, residuals, Ps


def perform_estimation(residuals, tracker):
    residuals = residuals - np.average(residuals, axis=0)
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(used_taps)
    R = estimate_noise_ukf_ml(correlation[0], tracker.Pz)
    truth = R_proto * (measurement_var + sim_var)
    error = matrix_error(R, truth)
    print("Truth:\n", truth)
    print("Estimation:\n", R)
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
    readings, filtered, residuals, Ps = filtering(sim, tracker)
    perform_estimation(residuals[-2*used_taps:], tracker)
    plot_filtered_values(readings, filtered, Ps)


if __name__ == "__main__":
    run_tracker()
