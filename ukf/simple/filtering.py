from __future__ import print_function
import numpy as np
from copy import copy
from math import sin, tan, cos
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from matplotlib import pyplot as plt
from noiseestimation.sensor import Sensor

# parameters
num_samples = 600
used_taps = num_samples / 2
dt = 0.1
measurement_var = 0.1
Q = 3e-5
var_pos = Q
var_heading = Q * 0.1
R_proto = np.array([[1, 0],
                    [0, 1]])
filter_misestimation_factor = 1
turning_threshold_angle = 0.001
wheelbase = 1


# move simulated robot
def f(x, u):
    x_onedim = False
    if len(x.shape) == 1:
        x_onedim = True
        x = x[..., np.newaxis]
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

    res = x + dx
    return res if not x_onedim else res[:, 0]


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
    sigmas = MerweScaledSigmaPoints(3, alpha=.1, beta=2., kappa=0)
    tracker = UKF(dim_x=3, dim_z=2, fx=f, hx=h, dt=dt, points=sigmas)
    tracker.Q = np.diag((var_pos, var_pos, var_heading))
    tracker.R = R_proto * measurement_var * filter_misestimation_factor
    tracker.x = np.array([0, 0, 0])
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * measurement_var] * num_samples
    readings, truths, filtered, residuals, Ps = [], [], [], [], []
    cmd = np.array([[1],
                    [0.1]])
    for R in Rs:
        sim.step(cmd)
        reading = sim.read(R)
        tracker.predict(cmd)
        tracker.update(reading[:, 0])
        readings.append(reading)
        truths.append(copy(sim.x))
        filtered.append(tracker.x)
        Ps.append(copy(tracker.P))
        residuals.append(tracker.y)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    Ps = np.asarray(Ps)
    residuals = np.asarray(residuals)
    return readings, truths, filtered, residuals, Ps


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
    readings, truths, filtered, residuals, Ps = filtering(sim, tracker)
    plot_results(readings, filtered, truths, Ps)


if __name__ == "__main__":
    run_tracker()
