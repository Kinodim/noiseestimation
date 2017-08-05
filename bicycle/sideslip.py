import numpy as np
from filterpy.kalman import KalmanFilter
from matplotlib import pyplot as plt
from noiseestimation.playback_sensor import PlaybackSensor

# parameters
measurement_var = 0.001
sim_var = 0.000
num_samples = 4000
# TODO change to a lower value and read measurements only when available
dt = 0.01

# constants for bicycle model
steering_to_wheel_angle = 0.067
cf = 156970  # cornering stiffness front [N/rad]
cb = 330000  # cornering stiffness back [N/rad]
lf = 1.45      # distance center of gravity (cog) <-> front axle
lb = 1.42      # distance cog <-> rear axle [m]
m = 2159       # vehicle mass [kg]
# radius of gyration, squared.
# This equals to J/m (J: moment of inertia) [m^2]
i_sq = 1.7601
J = m * i_sq   # moment of inertia


def generate_F_matrix(velocity):
    v = velocity

    # state: [beta, psi_d]
    f_11 = 1 - dt * (cf + cb) / m / v
    f_12 = -dt * (1 + (cf * lf - cb * lb) / m / v / v)
    f_21 = -dt * (cf * lf - cb * lb) / J
    f_22 = 1 - dt * (cf * lf * lf + cb * lb * lb) / J / v
    F = np.array([[f_11, f_12],
                  [f_21, f_22]])
    return F


def generate_B_matrix(velocity):
    v = velocity

    b_11 = dt * cf / m / v
    b_21 = dt * cf * lf / J
    B = np.array([[b_11],
                  [b_21]])
    return B


def setup():
    F = generate_F_matrix(velocity=0.001)
    H = np.array([[0, 1]])
    sim = PlaybackSensor("data/vehicle_state.json",
                         ["fVx", "fYawrate", "fStwAng"])
    # set up kalman filter
    tracker = KalmanFilter(dim_x=2, dim_z=1)
    tracker.F = F
    tracker.Q = np.eye(2) * 0.001
    tracker.H = H
    tracker.R = measurement_var
    tracker.x = np.array([[0, 0]]).T
    tracker.P = np.eye(2) * 500

    return sim, tracker


def filtering(sim, tracker):
    # perform sensor simulation and filtering
    Rs = [np.eye(1) * sim_var] * num_samples
    readings = []
    filtered = []
    for R in Rs:
        time, reading = sim.read(R)
        velocity = reading[0, 0]
        psi_d = reading[1, 0]
        delta = reading[2, 0] * steering_to_wheel_angle
        velocity = velocity if velocity > 0 else 0.1
        F = generate_F_matrix(velocity)
        B = generate_B_matrix(velocity)
        tracker.predict(u=delta, B=B, F=F)
        tracker.update(psi_d)
        readings.append(reading)
        filtered.append(tracker.x)

    readings = np.asarray(readings)
    filtered = np.asarray(filtered)
    return readings, filtered


def plot_results(readings, filtered):
    f, axarr = plt.subplots(2)
    axarr[0].set_title("Schwimmwinkel")
    axarr[0].plot(
        filtered[:, 0],
        'go')
    axarr[0].set_ylim((-0.3, 0.3))

    axarr[1].set_title("Gierrate")
    axarr[1].plot(
        filtered[:, 1],
        'ro')
    # axarr[1].plot(
    #     filtered[:, 0],
    #     filtered[:, 2],
    #     'm', linewidth=3)
    plt.show()


def run_tracker():
    sim, tracker = setup()
    readings, filtered = filtering(sim, tracker)
    plot_results(readings, filtered)


if __name__ == "__main__":
    run_tracker()
