import numpy as np
from numpy.linalg import pinv


def estimate_noise(C_arr, K, F, H, return_state_covariance=False):
    """An experimental variation of the mehra method
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
    return np.absolute(R)


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

    R = np.copy(G)
    if residual_type == "prior":
        R -= np.dot(H, np.dot(P, np.transpose(H)))
    elif residual_type == "posterior":
        R += np.dot(H, np.dot(P, np.transpose(H)))
    else:
        raise ValueError("Residual type %s not a valid option" % residual_type)
    return np.absolute(R)


def estimate_noise_extended(C_arr, K, F, H):
    """Estimates using adapted version for EKF with time-varying matrices

    Args:
        C_arr (ndarray): The list of innovation correlation estimates
        K (ndarray): Kalman gain
        F (ndarray): Update matrix
        H (ndarray): Measurement matrix

    Kalman gain, update matrix and the measurement matrix can be lists of those
    entities, containing the respective matrix for the according timestep
    starting with the oldest entry first

    Returns:
        ndarray: The estimated measurement noise covariance matrix
    """
    N = len(C_arr)

    # most recent entry first
    K = __reverse_or_create_list(K, N)
    F = __reverse_or_create_list(F, N)
    H = __reverse_or_create_list(H, N)

    # construct matrix A
    num_observations = H[0].shape[0]
    num_states = F[0].shape[0]
    A = np.ndarray((0, num_observations, num_states))
    # product = Product{F[n](I-K[n]H[n])}
    product = np.eye(num_states)
    for n in range(N - 1):
        if n != 0:
            # F[n] * (I - K[n]*H[n])
            bracket = np.dot(F[n], np.eye(num_states) - np.dot(K[n], H[n]))
            # watch out for order of multiplication: [0]...[n]
            product = np.dot(product, bracket)
        entry = np.dot(H[0], product)
        entry = np.dot(entry, F[0])
        A = np.vstack((A, [entry]))
    A = A.reshape((-1, num_states))

    C_stacked = C_arr[1:].reshape((-1, num_observations))
    MH = np.dot(K[0], C_arr[0]) + np.dot(pinv(A), C_stacked)
    R = C_arr[0] - np.dot(H[0], MH)
    return np.absolute(R)


def __reverse_or_create_list(x, N):
    if x.ndim == 3:
        x = x[::-1]
    else:
        x = np.asarray([x] * (N - 1))
    return x
