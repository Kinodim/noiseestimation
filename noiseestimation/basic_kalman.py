import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from matplotlib import pyplot as plt
from sensorsim import SensorSim

dt = 0.1
measurement_std_max = 4
measurement_std = np.arange(0, measurement_std_max, .05)
measurement_std = np.concatenate( (measurement_std, list(reversed(measurement_std)) ))
sim = SensorSim(measurement_std, timestep=dt, velocity = (0.5, 1))

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

readings = np.array([ np.array(sim.read()[0]).reshape(2,1) for _ in measurement_std ])

mu, cov, _, _ = tracker.batch_filter(readings)

plt.plot(readings[:,0], readings[:,1], 'go', label="Measurements")
plt.plot(mu[:,0], mu[:,2], 'm', linewidth=3, label="Filter")
plt.legend(loc="lower right")
plt.title("Kalman filtering of position")
plt.show()
