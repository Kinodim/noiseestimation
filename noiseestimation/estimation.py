import numpy as np
from numpy.linalg import pinv


def estimate_noise(C_arr, K, F, H, return_state_covariance=False):
    """Estimates noise based on the innovation correlation

    This function implements the approach proposed by Mehra.
    Using the matrices of a Kalman Filter and the innovation
    correlation, it calculates an estimation of the
    actual measurement noise covariance.
    For more information please refer to:
    On the Identification of Variances and Adaptive Kalman
    Filtering, Mehra 1970

    Args:
        C_arr (ndarray): The list of innovation correlation estimates
        K (ndarray): Kalman gain
        F (ndarray): Update matrix
        H (ndarray): Measurement matrix

    Returns:
        ndarray: The estimated measurement noise covariance matrix

    """
    N = len(C_arr)
    num_observations = H.shape[0]
    num_states = F.shape[0]

    # construct matrix A
    A = np.ndarray((0, num_observations, num_states))
    inner_expression = np.identity(num_states) - np.dot(K, H)
    inner_expression = np.dot(F, inner_expression)
    inner_bracket = np.identity(num_states)
    for n in range(N):
        if n != 0:
            inner_bracket = np.dot(inner_bracket, inner_expression)
        entry = np.dot(H, inner_bracket)
        entry = np.dot(entry, F)
        A = np.vstack((A, [entry]))

    A = A.reshape((-1, num_states))
    C_stacked = C_arr.reshape((-1, num_observations))
    MH = np.dot(K, C_arr[0]) + np.dot(pinv(A), C_stacked)
    R = C_arr[0] - np.dot(H, MH)
    if return_state_covariance:
        return R, MH
    else:
        return R


def estimate_noise_mehra(C_arr, K, F, H):
    """estimate using mehra version
    """
    N = len(C_arr)
    num_observations = H.shape[0]
    num_states = F.shape[0]

    # construct matrix A
    A = np.ndarray((0, num_observations, num_states))
    inner_expression = np.identity(num_states) - np.dot(K, H)
    inner_expression = np.dot(F, inner_expression)
    inner_bracket = np.identity(num_states)
    for n in range(N - 1):
        if n != 0:
            inner_bracket = np.dot(inner_bracket, inner_expression)
        entry = np.dot(H, inner_bracket)
        entry = np.dot(entry, F)
        A = np.vstack((A, [entry]))

    A = A.reshape((-1, num_states))
    C_stacked = C_arr[1:].reshape((-1, num_observations))
    MH = np.dot(K, C_arr[0]) + np.dot(pinv(A), C_stacked)
    R = C_arr[0] - np.dot(H, MH)
    return R


def estimate_noise_approx(G, H, P, residual_type="prior"):
    """Approximates noise based on the innovation variance

    This function implements another approach proposed by Mehra.
    Using the matrices of a Kalman Filter and the resulting
    innovation correlation, it calculates an estimation of the
    actual measurement noise covariance.
    For more information please refer to:
    Approaches to Adaptive Filtering, Mehra 1972
    Adaptive Kalman Filtering for INS/GPS, Mohamed and Schwarz 1999


    Args:
        G (ndarray): Estimate of the autocorrelation of the innovations
            with no time shift
        H (ndarray): Measurement matrix
        P (ndarray): Estimation covariance matrix
        residual_type (str): Type of passed innovations. A priori ('prior') or
            a posteriori ('posterior'). Default is 'prior'

    Returns:
        ndarray: The estimated measurement noise covariance matrix

    """

    R = G
    if residual_type == "prior":
        R -= np.dot(H, np.dot(P, np.transpose(H)))
    elif residual_type == "posterior":
        R += np.dot(H, np.dot(P, np.transpose(H)))
    else:
        raise ValueError("Residual type %s not a valid option" % residual_type)
    return R


def estimate_noise_extended(C_arr, K, F_arr, H_arr):
    """estimate using adapted version for EKF
    Args:
        C_arr (ndarray): The list of innovation correlation estimates
        K (ndarray): Kalman gain
        F_arr (ndarray): Update matrix list (starting with most recent)
        H_arr (ndarray): Measurement matrix list

    Returns:
        ndarray: The estimated measurement noise covariance matrix
    """
    N = len(C_arr)

    if H_arr.ndim == 3:
        A = __construct_A_nonlinear_H(N, K, F_arr, H_arr)
        num_observations = H_arr[0].shape[0]
    elif F_arr.ndim == 3:
        A = __construct_A_nonlinear_F(N, K, F_arr, H_arr)
        num_observations = H_arr.shape[0]

    C_stacked = C_arr[1:].reshape((-1, num_observations))
    MH = np.dot(K, C_arr[0]) + np.dot(pinv(A), C_stacked)
    R = C_arr[0] - np.dot(H_arr[0], MH)
    return R


def __construct_A_nonlinear_H(N, K, F, H_arr):
    num_observations = H_arr[0].shape[0]
    num_states = F.shape[0]

    A = np.ndarray((0, num_observations, num_states))
    # product = Product{F(I-KH[n])}
    product = np.eye(num_states)
    for n in range(N - 1):
        if n != 0:
            # F * (I - K*H[n])
            bracket = np.dot(F, np.eye(num_states) - np.dot(K, H_arr[n]))
            # watch out for order of multiplication: H[0]...H[n]
            product = np.dot(product, bracket)
        entry = np.dot(H_arr[0], product)
        entry = np.dot(entry, F)
        A = np.vstack((A, [entry]))

    A = A.reshape((-1, num_states))
    return A


def __construct_A_nonlinear_F(N, K, F_arr, H):
    num_observations = H.shape[0]
    num_states = F_arr[0].shape[0]

    A = np.ndarray((0, num_observations, num_states))
    # product = Product{F[n](I-KH)}
    product = np.eye(num_states)
    for n in range(N - 1):
        if n != 0:
            # F * (I - K*H[n])
            bracket = np.dot(F_arr[n], np.eye(num_states) - np.dot(K, H))
            # watch out for order of multiplication: F[0]...F[n]
            product = np.dot(product, bracket)
        entry = np.dot(H, product)
        entry = np.dot(entry, F_arr[0])
        A = np.vstack((A, [entry]))

    A = A.reshape((-1, num_states))
    return A
