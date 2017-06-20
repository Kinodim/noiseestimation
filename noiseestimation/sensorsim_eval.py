"""
Evaluate the SensorSim class by plotting the result of an increasing
and later decreasing measurement noise variance
"""
import numpy as np
from matplotlib import pyplot as plt
from sensorsim import SensorSim

measurement_std = np.arange(0, 4, .02)
measurement_std = np.concatenate( (measurement_std, list(reversed(measurement_std)) ))
sim = SensorSim(position=(0,0), velocity=(0.5,1), measurement_std=measurement_std, dim=2)

readings = np.array([np.array(sim.read()) for _ in measurement_std ])

plt.plot(readings[:,0, 0], readings[:,0, 1], 'go', label="sensor readings")
plt.plot(readings[:,1, 0], readings[:,1, 1], 'm', linewidth=2, label="true values")
plt.legend(loc='lower right')
plt.title("Position sensor readings")
plt.show()
