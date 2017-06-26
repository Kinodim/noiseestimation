import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from sensorsim import SensorSim

def run_tracker():
    # set up sensor simulator
    dt = 0.1
    measurement_std_max = 4
    measurement_std = np.arange(0, measurement_std_max, .02)
    measurement_std = np.concatenate( (measurement_std, list(reversed(measurement_std)) ))
    sim = SensorSim((0,0), (0.5,1), measurement_std, 2, timestep=dt)

    # set up kalman filter
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    tracker.Q = block_diag(q,q)
    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])
    tracker.R = np.diag([measurement_std_max, measurement_std_max])
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500

    # perform sensor simulation and filtering
    sensor_out = sim.batch_read()
    readings = sensor_out[:,0].reshape(-1,2,1)
    truth = sensor_out[:,1].reshape(-1,2,1)

    mu, cov, _, _ = tracker.batch_filter(readings)
    error = np.sqrt(np.sum(np.square(truth - mu[:,[0, 2]]), 1))

    # plot results
    f, axarr = plt.subplots(2)
    axarr[0].plot(readings[:,0], readings[:,1], 'go', label="Measurements")
    axarr[0].plot(mu[:,0], mu[:,2], 'm', linewidth=3, label="Filter")
    axarr[0].legend(loc="lower right")
    # axarr[0].set_xlim([0,200])
    axarr[0].set_title("Kalman filtering of position")

    axarr[1].plot(error, 'r')
    axarr[1].set_title("Estimation error")

    plt.show()

if __name__ == "__main__":
    run_tracker()
