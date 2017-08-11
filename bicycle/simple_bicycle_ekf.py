from filterpy.kalman import ExtendedKalmanFilter as EKF
import numpy as np
from math import sin, cos, tan


class SimpleBicycleEKF(EKF):
    turning_threshold_angle = 0.001

    def __init__(self, dt, wheelbase, var_vel, var_steer):
        """Derives the EKF class to implement the specific
        functioning of a mobile robot following a simple bicycle model.

        The state consists of the x and y position and the heading
        The control input contains the velocity and the steering angle
        The measurement is composed of a direct observation of the position
        """

        EKF.__init__(self, 3, 2, 2)
        self.dt = dt
        self.wheelbase = wheelbase
        self.var_vel = var_vel
        self.var_steer = var_steer

    def predict(self, u=0):
        self.move(u)
        F = self.Fx(u)
        B = self.Bx(u)

        # covariance of motion in control space
        M = np.diag((self.var_vel, self.var_steer))
        self.P = np.dot(F, np.dot(self.P, F.T)) + np.dot(B, np.dot(M, B.T))

    def update(self, z):
        def Hx(x):
            return np.dot(self.H, x)

        def HJacobian(x):
            return self.H

        EKF.update(self, z, HJacobian, Hx)

    def move(self, u):
        heading = self.x[2, 0]
        vel = u[0, 0]
        steering_angle = u[1, 0]
        dist = vel * self.dt

        # check whether robot is turning
        if abs(steering_angle) > self.turning_threshold_angle:
            beta = (dist / self.wheelbase) * tan(steering_angle)
            r = self.wheelbase / tan(steering_angle)
            dx = np.array([[-r * sin(heading) + r * sin(heading + beta)],
                           [r * cos(heading) - r * cos(heading + beta)],
                           [beta]])
        else:
            dx = np.array([[dist * cos(heading)],
                           [dist * sin(heading)],
                           [0]])

        self.x += dx

    def Fx(self, u):
        vel = u[0, 0]
        steering_angle = u[1, 0]
        theta = self.x[2, 0]
        if abs(steering_angle) < self.turning_threshold_angle:
            return np.eye(3)

        beta = (vel * self.dt / self.wheelbase) * tan(steering_angle)
        R = self.wheelbase / tan(steering_angle)
        return np.array(
            [[1, 0, -R*cos(theta) + R*cos(beta + theta)],
             [0, 1, -R*sin(theta) + R*sin(beta + theta)],
             [0, 0, 1]])

    def Bx(self, u):
        vel = u[0, 0]
        steering_angle = u[1, 0]
        theta = self.x[2, 0]
        # if abs(steering_angle) < self.turning_threshold_angle:
        #     return np.eye(3)

        d = vel * self.dt
        beta = (d / self.wheelbase) * tan(steering_angle)
        R = self.wheelbase / tan(steering_angle)
        alpha = steering_angle
        return np.array(
            [[self.dt*cos(beta + theta),
              -R*(-tan(alpha)**2 - 1)*sin(theta)/tan(alpha) + R*(-tan(alpha)**2 - 1)*sin(beta + theta)/tan(alpha) + d*(tan(alpha)**2 + 1)*cos(beta + theta)/tan(alpha)],
             [self.dt*sin(beta + theta),
              R*(-tan(alpha)**2 - 1)*cos(theta)/tan(alpha) - R*(-tan(alpha)**2 - 1)*cos(beta + theta)/tan(alpha) + d*(tan(alpha)**2 + 1)*sin(beta + theta)/tan(alpha)],
             [self.dt/R,
              d*(tan(alpha)**2 + 1)/self.wheelbase]])
