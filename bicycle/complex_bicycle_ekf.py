from filterpy.kalman import ExtendedKalmanFilter as EKF
import numpy as np


class ComplexBicycleEKF(EKF):
    minimum_velocity = 1e-3
    var_vel = 0.1
    var_steer = 0.01

    # Car parameters
    steering_to_wheel_angle = 0.067
    c_v = 156970  # cornering stiffness front [N/rad]
    c_h = 330000  # cornering stiffness back [N/rad]
    l_v = 1.45      # distance center of gravity (cog) <-> front axle
    l_h = 1.42      # distance cog <-> rear axle [m]
    m = 2159       # vehicle mass [kg]
    # radius of gyration squared,
    # this equals to J/m (J: moment of inertia) [m^2]
    i_sq = 1.7601
    J = m * i_sq   # moment of inertia

    def __init__(self, dt):
        """Derives the EKF class to implement the specific
        functioning of a mobile robot following a bicycle model.

        The state consists of the slip angle and yaw rate
        The control input contains current wheel angle and velocity
        The measurement is composed of a direct observation of the yaw rate
        """

        EKF.__init__(self, 2, 1, 2)
        self.dt = dt
        self.H = np.array([[0, 1]])

    def predict(self, u=0):
        u[1, 0] = max(self.minimum_velocity, u[1, 0])
        self.f_xu(u)
        F = self.Fx(u)
        self.F = F
        B = self.Bx(u)

        # covariance of motion in control space
        M = np.diag((self.var_steer, self.var_vel))
        self._P = np.dot(F, np.dot(self._P, F.T)) + np.dot(B, np.dot(M, B.T))

    def update(self, z):
        def Hx(x):
            return np.dot(self.H, x)

        def HJacobian(x):
            return self.H

        EKF.update(self, z, HJacobian, Hx)

    def f_xu(self, u):
        beta = self.x[0, 0]
        psi_d = self.x[1, 0]
        delta = u[0, 0] * self.steering_to_wheel_angle
        v = u[1, 0]

        self.x[0, 0] = (1 - self.dt * (self.c_v + self.c_h)
                        / self.m / v) * beta \
            - self.dt * (1 + (self.c_v * self.l_v - self.c_h * self.l_h)
                         / self.m / v**2) * psi_d \
            + self.dt * self.c_v / self.m / v * delta

        self.x[1, 0] = - self.dt * (self.c_v * self.l_v - self.c_h * self.l_h) \
            / self.J * beta \
            + (1 - self.dt * (self.c_v * self.l_v**2 + self.c_h * self.l_h**2)
               / self.J / v) * psi_d \
            + self.dt * self.c_v * self.l_v / self.J * delta

    def Fx(self, u):
        v = u[1, 0]
        F = np.array([[-self.dt*(self.c_h + self.c_v)/(self.m*v) + 1,
                       -self.dt*(1 + (-self.c_h*self.l_h + self.c_v*self.l_v)
                                 / (self.m * v**2))],
                      [-self.dt*(-self.c_h*self.l_h +
                                 self.c_v * self.l_v)/self.J,
                       1 - self.dt*(self.c_h*self.l_h**2 +
                                    self.c_v*self.l_v**2)/(self.J*v)]])
        return F

    def Bx(self, u):
        beta = self.x[0, 0]
        psi_d = self.x[1, 0]
        delta = u[0, 0] * self.steering_to_wheel_angle
        v = u[1, 0]
        B = np.array([[self.c_v*self.dt/(self.m*v),
                       beta*self.dt*(self.c_h + self.c_v)/(self.m*v**2) -
                       self.c_v*delta*self.dt/(self.m*v**2) +
                       2*self.dt*psi_d*(-self.c_h*self.l_h +
                                        self.c_v*self.l_v)/(self.m*v**3)],
                      [self.c_v*self.dt*self.l_v/self.J,
                       self.dt*psi_d*(self.c_h*self.l_h**2 +
                                      self.c_v*self.l_v**2)/(self.J*v**2)]])
        return B
