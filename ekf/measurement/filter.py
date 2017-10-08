import numpy as np
import math
from scipy.linalg import block_diag
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from noiseestimation.sensor import Sensor

# parameters
num_samples = 200
dt = 0.1
measurement_var_max = 0.1
R_proto = np.array([[1, 0],
                    [0, 0.2]])
filter_misestimation_factor = 1


# return range and bearing measurement
def h(x):
    distance = (x[0, 0]**2 + x[2, 0] ** 2) ** 0.5
    bearing = math.atan2(x[2, 0], x[0, 0])

    return np.array([[distance],
                     [bearing]])


def H_jacobian_at(x):
    hyp = x[0, 0]**2 + x[2, 0]**2
    dist = hyp ** 0.5
    return np.array(
        [[x[0, 0] / dist, 0., x[2, 0] / dist],
         [-x[2, 0] / hyp, 0., x[0, 0] / hyp]])


def normalize_angle(x):
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x


def custom_residual(a, b):
    res = np.array(
            [[a[0, 0] - b[0, 0]],
             [normalize_angle(a[1, 0] - b[1, 0])]])
    return res


def setup():
    # set up sensor simulator
    F = np.array([[1, dt, 0],
                  [0,  1, 0],
                  [0,  0, 1]])

    def f(x):
        return np.dot(F, x)

    x0 = np.array([[-5],
                   [0.5],
                   [1]])
    sim = Sensor(x0, f, h)

    # set up kalman filter
    tracker = ExtendedKalmanFilter(dim_x=3, dim_z=2)
    tracker.F = F
    q_x = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    q_y = 0.0001 * dt**2
    tracker.Q = block_diag(q_x, q_y)
    tracker.R = R_proto * measurement_var_max * filter_misestimation_factor
    tracker.x = np.array([[-5, 0.5, 1]]).T
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * measurement_var_max] * num_samples
    readings, truths, filtered, Ps = [], [], [], []
    for R in Rs:
        sim.step()
        reading = sim.read(R)
        tracker.predict()
        tracker.update(reading, H_jacobian_at, h, residual=custom_residual)
        readings.append(reading)
        truths.append(sim.x)
        filtered.append(tracker.x)
        Ps.append(tracker.P)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    Ps = np.asarray(Ps)
    return readings, truths, filtered, Ps


def plot_results(readings, filtered, truths, Ps):
    axarr = []
    axarr.append(plt.subplot(311))
    axarr[0].plot(
        filtered[:, 0],
        filtered[:, 2],
        'm', linewidth=3, label="Filter")
    axarr[0].plot(
        truths[:, 0],
        truths[:, 2],
        'k', linewidth=3, label="Truth")
    axarr[0].legend(loc="upper right")
    axarr[0].set_title("Kalman filtering of position")
    axarr[0].set_xlabel("x (m)")
    axarr[0].set_ylabel("y (m)")

    # confidences = np.sum(np.diagonal(np.square(Ps), axis1=1, axis2=2), axis=1)
    # axarr[1].plot(
    #     confidences,
    #     'b', linewidth=3)
    # axarr[1].set_title("State estimate covariance (P)")
    # axarr[1].set_ylim((0, 1))

    axarr.append(plt.subplot(312))
    axarr[1].plot(readings[:, 0, 0], 'go', label="Measurements")
    axarr[1].set_title("Range measurements")
    axarr[1].set_ylabel("Range (m)")

    axarr.append(plt.subplot(313, sharex=axarr[1]))
    axarr[2].plot(readings[:, 1, 0] * 180 / math.pi, 'go', label="Measurements")
    axarr[2].set_title("Bearing measurements")
    axarr[2].set_ylabel("Bearing (deg)")
    axarr[2].set_xlabel("Sample")

    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05,
                        hspace=0.25)
    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, truths, filtered, Ps = filtering(sim, tracker)
    plot_results(readings, filtered, truths, Ps)


if __name__ == "__main__":
    run_tracker()
