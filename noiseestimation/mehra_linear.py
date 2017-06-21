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
readings = []
truths = []
mu = []
residuals = []
for _ in measurement_std_list:
    reading, truth = sim.read()
    readings.extend(reading.flatten())
    truths.extend(truth.flatten())
    tracker.predict()
    tracker.update(reading)
    mu.extend(tracker.x[0])
    residuals.extend(tracker.y[0])

error = np.asarray(truths) - mu

# plot results
f, axarr = plt.subplots(3,sharex=True)
axarr[0].plot(readings, 'go', label="Measurements")
axarr[0].plot(mu, 'm', linewidth=3, label="Filter")
axarr[0].legend(loc="lower right")
# axarr[0].set_xlim([0,200])
axarr[0].set_title("Kalman filtering of position")

axarr[1].plot(error, 'r')
axarr[1].set_title("Estimation error")

axarr[2].plot(residuals, 'b')
axarr[2].set_title("Residuals")

plt.xlabel("Sample")
plt.show()
