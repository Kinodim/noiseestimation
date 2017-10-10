from __future__ import print_function
import copy
import numpy as np
from math import sin, tan, cos
from simple_bicycle_ekf import SimpleBicycleEKF
from matplotlib import pyplot as plt
from noiseestimation.sensor import Sensor

# parameters
num_samples = 250
dt = 0.1
measurement_var = 0.01
Q = 0.001
var_vel = Q
var_steer = .1 * Q
R_proto = np.array([[1, 0],
                    [0, 1]])
filter_misestimation_factor = 1
turning_threshold_angle = 0.001
wheelbase = 1


# move simulated robot
def f(x, u):
    heading = x[2, 0]
    vel = u[0, 0]
    steering_angle = u[1, 0]
    dist = vel * dt

    # check whether robot is turning
    if abs(steering_angle) > turning_threshold_angle:
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase / tan(steering_angle)
        dx = np.array([[-r * sin(heading) + r * sin(heading + beta)],
                       [r * cos(heading) - r * cos(heading + beta)],
                       [beta]])
    else:
        dx = np.array([[dist * cos(heading)],
                       [dist * sin(heading)],
                       [0]])

    return x + dx


def matrix_error(estimate, truth):
    return np.sqrt(np.sum(np.square(truth - estimate)))


def setup():
    # set up sensor simulator
    H = np.array([[1, 0, 0],
                  [0, 1, 0]])

    def h(x):
        return np.dot(H, x)

    x0 = np.array([[0],
                   [-1],
                   [0]])
    sim = Sensor(x0, f, h)

    # set up kalman filter
    tracker = SimpleBicycleEKF(dt, wheelbase, var_vel, var_steer)
    tracker.R = R_proto * measurement_var * filter_misestimation_factor
    tracker.x = np.array([[0, 0, 0]]).T
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    R = R_proto * measurement_var
    readings, truths, filtered, residuals, Ps, Fs, Ks = [], [], [], [], [], [], []
    cmds = [np.array([[0.6],
                      [0.23]])] * (num_samples / 4)
    cmds.extend([np.array([[2],
                           [-0.20]])] * (num_samples / 2))
    cmds.extend([np.array([[1],
                           [0.2]])] * (num_samples / 4))
    for cmd in cmds:
        sim.step(cmd)
        reading = sim.read(R)
        tracker.predict(cmd)
        tracker.update(reading)
        readings.append(reading)
        truths.append(sim.x)
        filtered.append(copy.copy(tracker.x))
        Ps.append(tracker.P)
        residuals.append(tracker.y)
        Fs.append(tracker.F)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    Ps = np.asarray(Ps)
    residuals = np.asarray(residuals)
    Fs = np.asarray(Fs)
    return readings, truths, filtered, residuals, Ps, Fs


def plot_results(readings, filtered, truths, Ps):
    axarr = [plt.subplot()]

    axarr[0].plot(
        readings[:, 0],
        readings[:, 1],
        'o', label="Readings"
    )
    axarr[0].plot(
        truths[:, 0],
        truths[:, 1],
        'k', linewidth=3, label="Truth")
    axarr[0].plot(
        filtered[:, 0],
        filtered[:, 1],
        linewidth=3, label="Filter")
    axarr[0].legend(loc="lower right")
    axarr[0].set_title("Kalman filtering of position")
    axarr[0].set_xlabel("x (m)")
    axarr[0].set_ylabel("y (m)")
    axarr[0].axis('scaled')

    # axarr[0].plot(Ps[:, 0, 0], 'g', label="X state variance")
    # axarr[0].plot(Ps[:, 1, 1], 'r', label="Y state variance")
    # axarr[0].legend(loc="upper right")
    # axarr[0].set_ylim((0, 0.002))
    # axarr[0].set_ylabel("$\sigma^2$ ($m^2$)")
    # axarr[0].set_xlabel("Sample")
    # axarr[0].set_title("State covariance")

    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, truths, filtered, residuals, Ps, Fs = filtering(sim, tracker)
    print(np.sqrt(np.sum(np.square(truths-filtered)[50:,[0,1],0])))
    plot_results(readings, filtered, truths, Ps)


if __name__ == "__main__":
    run_tracker()
