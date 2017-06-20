import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from sensorsim import SensorSim

# set up sensor simulator
dt = 0.1
measurement_std = 2
measurement_std_list = np.asarray([measurement_std] * 1000)
sim = SensorSim(0, 0.1, measurement_std_list, 1, timestep=dt)

# set up kalman filter
tracker = KalmanFilter(dim_x=2, dim_z=1)
tracker.F = np.array([[1, dt],
                      [0,  1]])
q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
tracker.Q = q
tracker.H = np.array([[1, 0]])
tracker.R = measurement_std
tracker.x = np.array([[0, 0]]).T
tracker.P = np.eye(2) * 500

# perform sensor simulation and filtering
sensor_out = sim.batch_read()
readings = sensor_out[:,0].reshape(-1,1,1)
truth = sensor_out[:,1].reshape(-1,1,1)

mu, cov, _, _ = tracker.batch_filter(readings)
error = np.square(truth[:,0] - mu[:,0])

# plot results
f, axarr = plt.subplots(2)
axarr[0].plot(readings[:,0], 'go', label="Measurements")
axarr[0].plot(mu[:,0], 'm', linewidth=3, label="Filter")
axarr[0].legend(loc="lower right")
# axarr[0].set_xlim([0,200])
axarr[0].set_title("Kalman filtering of position")

axarr[1].plot(error[:,0], 'r')
axarr[1].set_title("Estimation error")

plt.show()
