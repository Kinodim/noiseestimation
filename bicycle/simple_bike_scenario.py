import numpy as np
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor
from simple_bike_ekf import SimpleBikeEKF

# Parameters
num_samples = 11650
steering_to_wheel_angle = 0.067
lf = 1.45      # distance center of gravity (cog) <-> front axle
lb = 1.42      # distance cog <-> rear axle [m]
wheelbase = lf + lb
var_vel = 0.001
var_steer = 0.0001
var_measurement = 0.01
var_sim = 0.0001
dt = 0.01


def setup():
    sim = PlaybackSensor("data/vehicle_state_integrated.json",
                         ["fVx", "fStwAng", "pose_x", "pose_y"])
    # set up kalman filter
    tracker = SimpleBikeEKF(dt, wheelbase, var_vel, var_steer)
    tracker.Q = np.eye(2) * 0.001
    tracker.H = np.array([[1, 0, 0],
                          [0, 1, 0]])
    tracker.R = var_measurement
    tracker.x = np.array([[0, 0, 0]]).T
    tracker.P = np.eye(3) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    readings = []
    filtered = []
    R = [[var_sim]]
    for _ in range(num_samples):
        time, reading = sim.read(R)
        velocity = reading[0, 0]
        delta = reading[1, 0] * steering_to_wheel_angle
        velocity = velocity if velocity > 0 else 0.1
        tracker.predict(u=np.array([[velocity],
                                    [delta]]))
        tracker.update(reading[2:])
        readings.append(reading)
        filtered.append(tracker.x)

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    return readings, filtered


def plot_results(readings, filtered):
    plt.plot(
        filtered[:, 0],
        filtered[:, 1],
        "b-"
    )
    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, filtered = filtering(sim, tracker)
    plot_results(readings, filtered)


if __name__ == "__main__":
    run_tracker()
