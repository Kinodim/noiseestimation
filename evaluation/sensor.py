import numpy as np
from matplotlib import pyplot as plt
from noiseestimation.sensor import LinearSensor

dt = 0.1
F = np.array([[1, dt, 0,  0],
              [0,  1, 0,  0],
              [0,  0, 1, dt],
              [0,  0, 0,  1]])
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
x0 = np.array([[0],
               [0.5],
               [0],
               [1]])
sim = LinearSensor(x0, F, H)

measurement_vars = np.linspace(0, 1, 500)
Rs = [np.eye(2) * measurement_var for measurement_var in measurement_vars]
readings = []
for R in Rs:
    sim.step()
    reading = sim.read(R)
    readings.append(reading)
readings = np.asarray(readings)

plt.plot(
    readings[:, 0],
    readings[:, 1],
    'go', label="Measurements")
plt.show()
