import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from sensorsim import SensorSim
from correlator import Correlator
from noiseestimator import estimate_noise
from math import fabs


def run_tracker():
    # set up sensor simulator
    dt = 0.1

    measurement_std = 1
    measurement_std_list = np.asarray([measurement_std] * 200)
    sim = SensorSim((0,0), (0.5,1), measurement_std_list, 2, timestep=dt)

    # set up kalman filter
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)
    tracker.Q = block_diag(q,q)
    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])
    tracker.R = np.diag([measurement_std, measurement_std])
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500


    # perform sensor simulation and filtering
    readings = []
    truths = []
    mu = []
    residuals = []
    for _ in measurement_std_list:
        reading, truth = sim.read()
        readings.append(reading)
        truths.append(truth)
        tracker.predict()
        tracker.update(reading)
        mu.append( tracker.x[(0,2), :] )
        residuals.append(tracker.y)

    readings = np.asarray(readings)
    mu = np.asarray(mu)
    truths = np.asarray(truths)
    error = np.sqrt(np.sum(np.square(truths - mu), 1))

    # plot results
    f, axarr = plt.subplots(2)
    axarr[0].plot(readings[:,0], readings[:,1], 'go', label="Measurements")
    axarr[0].plot(mu[:,0], mu[:,1], 'm', linewidth=3, label="Filter")
    axarr[0].legend(loc="lower right")
    # axarr[0].set_xlim([0,200])
    axarr[0].set_title("Kalman filtering of position")

    axarr[1].plot(error, 'r')
    axarr[1].set_title("Estimation error")

    plt.show()

    # plot results
    # f, axarr = plt.subplots(3,sharex=True)
    # axarr[0].plot(readings, 'go', label="Measurements")
    # axarr[0].plot(mu, 'm', linewidth=3, label="Filter")
    # axarr[0].legend(loc="lower right")
    # axarr[0].set_xlim([0,200])
    # axarr[0].set_title("Kalman filtering of position")

    # axarr[1].plot(error, 'r')
    # axarr[1].set_title("Estimation error")

    # axarr[2].plot(residuals, 'b')
    # axarr[2].set_title("Residuals")

    # cor = Correlator(residuals[-50:])
    # print cor.isWhite()
    # cor = Correlator(residuals)
    # print cor.isWhite()

    # plt.show()

    # cor = Correlator(residuals)
    # R = estimate_noise(cor.covariance(50), tracker.K, tracker.F, tracker.H)
    # abs_err = measurement_std**2 - R
    # rel_err = abs_err / measurement_std**2
    # print "True: %.3f" % measurement_std**2
    # print "Filter: %.3f" % tracker.R
    # print "Estimated: %.3f" % R
    # print "Absolute error: %.3f" % abs_err
    # print "Relative error: %.3f %%" % (rel_err * 100)
    # print "-" * 15
    # return rel_err

if __name__ == "__main__":
    run_tracker()
    # sum = .0
    # runs = 100
    # for i in range(runs):
    #     print "%d / %d" % (i+1, runs)
    #     sum += fabs(run_tracker())

    # print "Avg relative error: %.3f %%" % (sum * 100 / runs)
