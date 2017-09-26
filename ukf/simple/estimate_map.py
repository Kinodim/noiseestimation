from __future__ import print_function
import numpy as np
from copy import copy
from math import sin, tan, cos
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from matplotlib import pyplot as plt
from noiseestimation.sensor import Sensor
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import estimate_noise_ukf_map

# parameters
num_samples = 600
used_taps = 50
dt = 0.1
measurement_var = 0.02
var_pos = 5e-4
var_heading = 1e-5
R_proto = np.array([[2, 0],
                    [0, 2]])
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
    sigmas = MerweScaledSigmaPoints(3, alpha=.1, beta=2., kappa=1)
    tracker = UKF(dim_x=3, dim_z=2, fx=f, hx=h, dt=dt, points=sigmas)
    tracker.Q = np.diag((var_pos, var_pos, var_heading))
    tracker.R = R_proto * measurement_var * filter_misestimation_factor
    tracker.x = np.array([0, 0, 0])
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * measurement_var] * num_samples
    readings, truths, filtered, Ps, estimated_Rs = [], [], [], [], [0]
    cmd = np.array([[1],
                    [0.1]])
    for index, R in enumerate(Rs):
        sim.step(cmd)
        reading = sim.read(R)
        tracker.predict(cmd)
        tracker.update(reading[:, 0])
        readings.append(reading)
        truths.append(sim.x)
        filtered.append(copy(tracker.x))
        Ps.append(copy(tracker.P))
        residual = tracker.y[:, np.newaxis]
        starting_index = 100
        if index > starting_index:
            R_estimation = perform_estimation(residual, tracker,
                                              index - starting_index,
                                              estimated_Rs[-1])
            estimated_Rs.append(R_estimation)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    Ps = np.asarray(Ps)
    return readings, truths, filtered, Ps, estimated_Rs[-1]


def perform_estimation(residual, tracker, index, old_estimate):
    average_factor = (1 - 0.95) / (1 - 0.95**(index + 1))
    estimate = estimate_noise_ukf_map(residual, tracker.Pz, average_factor,
                                      old_estimate, False)
    return estimate


def evaluate_estimation(estimation):
    truth = R_proto * measurement_var
    error = matrix_error(estimation, truth)
    print("Truth:\n", truth)
    print("Estimation:\n", estimation)
    print("Error: %.6f" % error)
    print("-" * 15)
    return error


def plot_results(readings, filtered, truths, Ps):
    f, axarr = plt.subplots(2)
    axarr[0].plot(
        readings[:, 0],
        readings[:, 1],
        'go', label="Readings"
    )
    axarr[0].plot(
        truths[:, 0],
        truths[:, 1],
        'k', linewidth=3, label="Truth")
    axarr[0].plot(
        filtered[:, 0],
        filtered[:, 1],
        'm', linewidth=3, label="Filter")
    axarr[0].legend(loc="lower right")
    axarr[0].set_title("Kalman filtering of position")
    # axarr[0].axis('scaled')

    axarr[1].plot(Ps[:, 0, 0], label="X Variance")
    axarr[1].plot(Ps[:, 1, 1], label="Y Variance")
    axarr[1].legend(loc="upper right")

    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, truths, filtered, Ps, estimation = filtering(sim, tracker)
    evaluate_estimation(estimation)
    plot_results(readings, filtered, truths, Ps)


if __name__ == "__main__":
    run_tracker()
