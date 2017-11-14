from __future__ import print_function
import numpy as np
from copy import copy
from math import sin, tan, cos
from matplotlib import pyplot as plt
from filterpy.kalman import MerweScaledSigmaPoints
from noiseestimation.UKF import EstimationUnscentedKalmanFilter as UKF
from noiseestimation.sensor import Sensor
from noiseestimation.correlator import Correlator
from noiseestimation.estimation import (
    estimate_noise_ukf_ml,
    estimate_noise_ukf_map,
    estimate_noise_ukf_scaling
)

# parameters
skip_initial = 100
num_samples = 300
used_taps = num_samples / 2
dt = 0.1
measurement_var = 0.1
Q = 3e-5
var_pos = Q
var_heading = Q * 0.1
R_proto = np.array([[1, 0.1],
                    [0.1, 1.5]])
filter_misestimation_factor = 1
map_b = 0.9999

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
    sigmas = MerweScaledSigmaPoints(3, alpha=.1, beta=2., kappa=0)
    tracker = UKF(dim_x=3, dim_z=2, fx=f, hx=h, dt=dt, points=sigmas)
    tracker.Q = np.diag((var_pos, var_pos, var_heading))
    tracker.R = R_proto * measurement_var * filter_misestimation_factor
    tracker.x = np.array([0, 0, 0])
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [R_proto * measurement_var] * (num_samples + skip_initial)
    (readings, truths, filtered, residuals, Ps,
     map_estimations, map_estimations_convergence) = (
        [], [], [], [], [], [0], [0])
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
        residuals.append(residual)
        starting_index = skip_initial
        if index > starting_index:
            average_factor = (1 - map_b) / (1 -
                                           map_b**(index - starting_index))
            estimate = estimate_noise_ukf_map(
                residual, tracker.Pz, average_factor,
                map_estimations[-1], False)
            estimate_convergence = estimate_noise_ukf_map(
                residual, tracker.Pz, average_factor,
                map_estimations_convergence[-1], True)
            map_estimations.append(estimate)
            map_estimations_convergence.append(estimate_convergence)

    readings = np.asarray(readings)
    truths = np.asarray(truths)
    filtered = np.asarray(filtered)
    Ps = np.asarray(Ps)
    residuals = np.asarray(residuals)
    return (readings, truths, filtered, residuals, Ps,
            map_estimations[-1], map_estimations_convergence[-1])


def perform_estimation(residuals, tracker,
                       map_estimate, map_estimate_convergence):
    cor = Correlator(residuals)
    correlation = cor.autocorrelation(used_taps)
    R_ml = estimate_noise_ukf_ml(correlation[0], tracker.Pz)
    R_scaled = estimate_noise_ukf_scaling(correlation[0], tracker.Pz, tracker.R)
    R_map = map_estimate
    R_map_conv = map_estimate_convergence
    truth = R_proto * measurement_var
    truth_norm = matrix_error(truth, 0)
    error_ml = matrix_error(R_ml, truth)
    error_scaled = matrix_error(R_scaled, truth)
    error_map = matrix_error(R_map, truth)
    error_map_conv = matrix_error(R_map_conv, truth)
    print("Truth")
    print(truth)
    print("ML:")
    print("", R_ml)
    print("\tRelative error: %.6f" % (error_ml / truth_norm))
    print("Scaled:")
    print("", R_scaled)
    print("\tRelative error: %.6f" % (error_scaled / truth_norm))
    print("MAP:")
    print("", R_map)
    print("\tRelative error: %.6f" % (error_map / truth_norm))
    return error_ml, error_scaled, error_map, error_map_conv


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


def run_tracker(dummy):
    sim, tracker = setup()
    (readings, truths, filtered, residuals, Ps,
     map_estimate, map_estimate_convergence) = filtering(sim, tracker)
    errors = perform_estimation(residuals[skip_initial:], tracker,
                                map_estimate, map_estimate_convergence)
    # plot_results(readings, filtered, truths, Ps)
    return errors


if __name__ == "__main__":
    errors = run_tracker(0)
