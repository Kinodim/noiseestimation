import numpy as np
from numpy.linalg import pinv

def estimate_noise(C_arr, K, F, H):
    """Estimates noise based on the innovation correlation

    This function implements the approach proposed by Mehra.
    Using the matrices of a Kalman Filter and the innovation
    covariance, it calculates an estimation of the
    actual measurement noise covariance.
    For more information please refer to:
    On the Identification of Variances and Adaptive Kalman
    Filtering, Mehra 1970

    Args:
        C_arr (ndarray): The list of innovation covariance estimates
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
    inner_expression = np.identity(num_states) - np.dot(K,H)
    inner_expression = np.dot(F,inner_expression)
    inner_bracket = np.identity(num_states)
    for n in range(N):
        if n != 0:
            inner_bracket = np.dot(inner_bracket, inner_expression)
        entry = np.dot(H, inner_bracket)
        entry = np.dot(entry, F)
        A = np.vstack( (A, [entry]) )

    A = A.reshape( (-1, num_states))
    C_stacked = C_arr.reshape( (-1, num_observations))
    MH = np.dot(K, C_arr[0]) + np.dot( pinv(A), C_stacked)
    R = C_arr[0] - np.dot(H, MH)
    return R

def estimate_noise_mehra(C_arr, K, F, H):
    """estimate using mehra version
    """
    N = len(C_arr)
    num_observations = H.shape[0]
    num_states = F.shape[0]

    # construct matrix A
    A = np.ndarray((0, num_observations, num_states))
    inner_expression = np.identity(num_states) - np.dot(K,H)
    inner_expression = np.dot(F,inner_expression)
    inner_bracket = np.identity(num_states)
    for n in range(N - 1):
        if n != 0:
            inner_bracket = np.dot(inner_bracket, inner_expression)
        entry = np.dot(H, inner_bracket)
        entry = np.dot(entry, F)
        A = np.vstack( (A, [entry]) )

    A = A.reshape( (-1, num_states))
    C_stacked = C_arr[1:].reshape( (-1, num_observations))
    MH = np.dot(K, C_arr[0]) + np.dot( pinv(A), C_stacked)
    R = C_arr[0] - np.dot(H, MH)
    return R

def estimate_noise_approx(G, H, P):
    """Approximates noise based on the innovation variance

    This function implements another approach proposed by Mehra.
    Using the matrices of a Kalman Filter and the resulting
    innovation covariance, it calculates an estimation of the
    actual measurement noise covariance.
    For more information please refer to:
    Approaches to Adaptive Filering, Mehra 1972

    Args:
        G (ndarray): Estimate of the autocorrelation of the innovations
            with no time shift
        H (ndarray): Measurement matrix
        P (ndarray): Estimation covariance matrix

    Returns:
        ndarray: The estimated measurement noise covariance matrix

    """

    R = G - np.dot(H, np.dot(P, np.transpose(H)))
    return R
