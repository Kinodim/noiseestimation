import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from noiseestimation.sensor import LinearSensor

# parameters
measurement_var_max = 0.1
num_samples = 600


def setup():
    # set up sensor simulator
    dt = 0.1
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    x0 = np.array([[0],
                   [0.5],
                   [0],
                   [1]])
    sim = LinearSensor(x0, F, H)

    # set up kalman filter
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    tracker.F = F
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    tracker.Q = block_diag(q, q)
    tracker.H = H
    tracker.R = np.diag([measurement_var_max, measurement_var_max])
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    measurement_vars = np.linspace(0, measurement_var_max, num_samples / 2)
    measurement_vars = np.concatenate(
        (measurement_vars, list(reversed(measurement_vars))))
    Rs = [np.eye(2) * measurement_var for measurement_var in measurement_vars]
    Rs = [np.eye(2) * measurement_var_max] * (num_samples / 4)
    readings = []
    truths = []
    filtered = []
    for R in Rs:
        sim.step()
        reading = sim.read(R)
        tracker.predict()
        tracker.update(reading)
        readings.append(reading)
        truths.append(sim.x)
        filtered.append(tracker.x)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    error = np.sqrt(np.sum(
        np.square(truths[:, [0, 2]] - filtered[:, [0, 2]]),
        1))
    return readings, truths, filtered, error


def plot_results(readings, filtered, error):
    f, axarr = plt.subplots(1)
    axarr.plot(
        readings[:, 0],
        readings[:, 1],
        'go', label="Measurements")
    axarr.plot(
        filtered[:, 0],
        filtered[:, 2],
        'm', linewidth=3, label="Filter")
    axarr.legend(loc="lower right")
    axarr.set_xlabel("x (m)")
    axarr.set_ylabel("y (m)")
    axarr.set_title("Kalman filtering of position")

    # axarr[1].plot(error, 'r')
    # axarr[1].set_title("Estimation error")
    # axarr[1].set_xlabel("Sample")
    # axarr[1].set_ylabel("Error (m)")
    # axarr[1].set_ylim((0, 0.65))

    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, truths, filtered, error = filtering(sim, tracker)
    plot_results(readings, filtered, error)


if __name__ == "__main__":
    run_tracker()
