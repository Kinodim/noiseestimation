import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF


def steering_to_wheel_angle(angle):
    min_steering_wheel_angle = -8.737119
    max_steering_wheel_angle = 8.676032
    steering_to_front_values = [-0.628560, -0.614293, -0.600124, -0.586052,
                                -0.572073, -0.558187, -0.544392, -0.530684,
                                -0.517063, -0.503527, -0.490072, -0.476699,
                                -0.463404, -0.450185, -0.437041, -0.423969,
                                -0.410968, -0.398036, -0.385170, -0.372370,
                                -0.359631, -0.346954, -0.334335, -0.321774,
                                -0.309267, -0.296812, -0.284409, -0.272055,
                                -0.259748, -0.247485, -0.235266, -0.223088,
                                -0.210948, -0.198846, -0.186779, -0.174745,
                                -0.162743, -0.150769, -0.138823, -0.126902,
                                -0.115005, -0.103128, -0.091272, -0.079432,
                                -0.067608, -0.055797, -0.043998, -0.032208,
                                -0.020426, -0.008650, 0.003123, 0.014895,
                                0.026666, 0.038440, 0.050219, 0.062003,
                                0.073796, 0.085599, 0.097415, 0.109244,
                                0.121090, 0.132953, 0.144837, 0.156743,
                                0.168672, 0.180628, 0.192612, 0.204625,
                                0.216671, 0.228750, 0.240865, 0.253018,
                                0.265211, 0.277445, 0.289723, 0.302047,
                                0.314419, 0.326840, 0.339313, 0.351839,
                                0.364421, 0.377061, 0.389760, 0.402520,
                                0.415344, 0.428233, 0.441190, 0.454215,
                                0.467313, 0.480483, 0.493729, 0.507052,
                                0.520454, 0.533938, 0.547504, 0.561156,
                                0.574895, 0.588723, 0.602641, 0.616653]
    steering_values = np.linspace(min_steering_wheel_angle,
                                  max_steering_wheel_angle,
                                  len(steering_to_front_values))
    return np.interp(angle, steering_values, steering_to_front_values)


class BicycleUKF(UKF):
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

    # Sigma points parameters
    alpha = 1e-1
    beta = 2.
    kappa = 0.1

    def __init__(self, dt):
        """Derives the UKF class to implement the specific
        functioning of a mobile robot following a bicycle model.

        The state consists of the slip angle, yaw rate and velocity
        The control input contains current wheel angle and acceleration
        The measurement is composed of an observation of the yaw rate and
        velocity
        """
        self.dt = dt
        H = np.array([[0, 1, 0],
                      [0, 0, 1]])

        def h(x):
            return np.dot(H, x)

        sigmas = MerweScaledSigmaPoints(3, alpha=self.alpha, beta=self.beta,
                                        kappa=self.kappa)
        UKF.__init__(self, dim_x=3, dim_z=2,
                     fx=self.f_xu, hx=h, dt=dt, points=sigmas)

    def f_xu(self, x, dt, u):
        beta = x[0]
        psi_d = x[1]
        v = x[2]
        delta = steering_to_wheel_angle(u[0, 0])
        a = u[1, 0]

        res = np.zeros((3))
        res[0] = beta*(-self.dt*(self.c_h + self.c_v)/(self.m*v) + 1) + \
            self.c_v*delta*self.dt/(self.m*v) - \
            self.dt*psi_d*(1 + (-self.c_h*self.l_h + self.c_v*self.l_v)
                           / (self.m * v**2))
        res[1] = psi_d*(1 - self.dt*(self.c_h*self.l_h**2 +
                                     self.c_v*self.l_v**2)/(self.J*v)) - \
            beta*self.dt*(-self.c_h*self.l_h + self.c_v*self.l_v)/self.J + \
            self.c_v*delta*self.dt*self.l_v/self.J
        res[2] = a*self.dt + v
        return res
