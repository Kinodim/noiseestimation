import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from noiseestimation.sensorsim import SensorSim
from noiseestimation.correlator import Correlator
from noiseestimation.noiseestimator import estimate_noise, estimate_noise_approx


def plot_results(readings, mu, Rs, measurement_std_list):
    f, axarr = plt.subplots(2)
    axarr[0].plot(readings, 'go', label="Measurements")
    axarr[0].plot(mu, 'm', linewidth=3, label="Filter")
    axarr[0].legend(loc="lower right")
    # axarr[0].set_xlim([0,200])
    axarr[0].set_title("Kalman filtering of position")

    axarr[1].plot(Rs, 'y', label="Estimate")
    axarr[1].plot([x ** 2 for x in measurement_std_list], 'b', label="Truth")
    axarr[1].set_title("Noise covariance estimaton")
    axarr[1].legend(loc="upper right")

    plt.show()


def run_tracker():
    # parameters
    measurement_std = 3.5
    filter_misestimation_factor = 1.0
    sample_size = 2000
    estimation_sample_size = 80
    used_taps = int(estimation_sample_size * 0.5)

    # set up sensor simulator
    dt = 0.1
    # measurement_std_list = np.asarray([measurement_std] * sample_size)
    measurement_std_list = np.linspace(
        measurement_std / 5,
        measurement_std * 1,
        sample_size / 2)
    measurement_std_list = np.concatenate(
        (measurement_std_list, list(reversed(measurement_std_list))))
    sim = SensorSim(0, 0.1, measurement_std_list, 1, timestep=dt)

    # set up kalman filter
    tracker = KalmanFilter(dim_x=2, dim_z=1)
    tracker.F = np.array([[1, dt],
                          [0,  1]])
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)
    tracker.Q = q
    tracker.H = np.array([[1, 0]])
    tracker.R = measurement_std**2 * filter_misestimation_factor
    tracker.x = np.array([[0, 0]]).T
    tracker.P = np.eye(2) * 500

    # perform sensor simulation and filtering
    readings = []
    truths = []
    mu = []
    residuals = []
    Rs = []
    for idx, _ in enumerate(measurement_std_list):
        reading, truth = sim.read()
        readings.extend(reading.flatten())
        truths.extend(truth.flatten())
        tracker.predict()
        tracker.update(reading)
        mu.extend(tracker.x[0])
        residuals.extend(tracker.y[0])
        Rs.append(tracker.R)

        if (idx < estimation_sample_size or
                idx % (estimation_sample_size / 10) != 0):
            print(idx)
            continue
        cor = Correlator(residuals[-estimation_sample_size:])
        used_taps = int(estimation_sample_size / 2)
        correlation = cor.autocorrelation(used_taps)
        R = estimate_noise(correlation, tracker.K, tracker.F, tracker.H)
        R_approx = estimate_noise_approx(correlation[0], tracker.H, tracker.P)
        abs_err = measurement_std**2 - R
        rel_err = abs_err / measurement_std**2
        print("True: %.3f" % measurement_std**2)
        print("Filter: %.3f" % tracker.R)
        print("Estimated: %.3f" % R)
        print("Estimated (approximation): %.3f" % R_approx)
        print("Absolute error: %.3f" % abs_err)
        print("Relative error: %.3f %%" % (rel_err * 100))
        print("-" * 15)
        if(R > 0):
            tracker.R = R
        # if(R_approx > 0):
        #     tracker.R = R_approx

    # error = np.asarray(truths) - mu

    plot_results(readings, mu, Rs, measurement_std_list)


if __name__ == "__main__":
    run_tracker()
