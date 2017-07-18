import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from noiseestimation.sensor import Sensor

# parameters
measurement_var_max = 0.001
num_samples = 600
dt = 0.1


# return range measurement
def h(x):
    return (x[0, 0]**2 + x[2, 0] ** 2) ** 0.5


def H_jacobian_at(x):
    denom = (x[0, 0]**2 + x[2, 0]**2) ** 0.5
    return np.array([[x[0, 0] / denom, 0., x[2, 0] / denom]])


def setup():
    # set up sensor simulator
    F = np.array([[1, dt, 0],
                  [0,  1, 0],
                  [0,  0, 1]])

    def f(x):
        return np.dot(F, x)

    x0 = np.array([[-10],
                   [0.4],
                   [5]])
    sim = Sensor(x0, f, h)

    # set up kalman filter
    tracker = ExtendedKalmanFilter(dim_x=3, dim_z=1)
    tracker.F = F
    q_x = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    q_y = 0.001 * dt**2
    tracker.Q = block_diag(q_x, q_y)
    tracker.R = measurement_var_max
    tracker.x = np.array([[-10.1, 0.4, 5.1]]).T
    tracker.P = np.eye(3) * 1

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    measurement_vars = np.linspace(0, measurement_var_max, num_samples / 2)
    measurement_vars = np.concatenate(
        (measurement_vars, list(reversed(measurement_vars))))
    Rs = [np.eye(2) * measurement_var for measurement_var in measurement_vars]
    Rs = [[[measurement_var_max]]] * num_samples
    readings = []
    truths = []
    filtered = []
    for R in Rs:
        sim.step()
        reading = sim.read(R)
        tracker.predict()
        tracker.update(reading, H_jacobian_at, h)
        readings.append(reading)
        truths.append(sim.x)
        filtered.append(tracker.x)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    # error = np.sqrt(np.sum(
    #     np.square(truths[:, [0, 2]] - filtered[:, [0, 2]]),
    #     1))
    error = []
    return readings, truths, filtered, error


def plot_results(readings, filtered, error):
    f, axarr = plt.subplots(3)
    axarr[0].plot(
        filtered[:, 0],
        filtered[:, 2],
        'm', linewidth=3, label="Filter")
    axarr[0].legend(loc="lower right")
    axarr[0].set_title("Kalman filtering of position")

    axarr[1].plot(
        filtered[:, 1],
        'b', linewidth=3, label="Filter")
    axarr[1].legend(loc="lower right")
    axarr[1].set_title("Kalman filtering of v_x")

    axarr[2].plot(readings[:, 0, 0], 'go', label="Measurements")
    axarr[2].set_title("Range measurements")

    # axarr[1].plot(error, 'r')
    # axarr[1].set_title("Estimation error")

    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, truths, filtered, error = filtering(sim, tracker)
    plot_results(readings, filtered, error)


if __name__ == "__main__":
    run_tracker()
