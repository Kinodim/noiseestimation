from math import sqrt, ceil
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np


class Correlator:
    """
    Provides functionality related to the autocorrelation coefficients
    of a sequence of values, more specifically to check whether the
    sequence is some form of white noise.

    Two options to check this are offered:
    The first is to use its estimated autocorrelation
    and the times they exceed the 95% confidence interval.
    This approach is adapted from: On the Identification of Variances
    and Adaptive Kalman Filtering, Mehra 1970

    The alternative is to employ the Ljung-Box test implemented in the
    statsmodel package
    """

    def __init__(self, values):
        self.values = np.asarray(values)
        # convert to required format in case we receive simple list
        # shape from (n,) to (n, 1, 1)
        if len(self.values.shape) == 1:
            self.values = self.values[:, np.newaxis, np.newaxis]

    def covariance(self, lags):
        C = []
        for k in range(lags + 1):
            c = np.zeros((self.values.shape[1], self.values.shape[1]))
            for i in range(k, len(self.values)):
                c += np.dot(self.values[i], np.transpose(self.values[i-k]))
            c /= len(self.values)
            C.append(c)
        return np.asarray(C)

    def autocorrelation(self, lags):
        C = self.covariance(lags)
        C_0_diagonals = np.diagonal(C[0]).reshape((-1, 1))
        denominator = np.sqrt(
            np.dot(
                C_0_diagonals,
                np.transpose(C_0_diagonals)))
        rho = [c / denominator for c in C]
        return np.asarray(rho)

    def isWhite(self, method='ljung-box', lags=0):
        """Checks whether the passed sequence is white noise

        Args:
            method (str, optional): Uses statistical method to employ.
                Available options: "ljung-box" (default) and "mehra"
            lags (int): number of taps to use for autocorrelation

        Returns:
            bool: True if sequence is white
        """
        # TODO extend to multiple dimensions
        if method == 'mehra':
            limit = 1.96 / sqrt(len(self.values))
            outliers = 0

            if lags == 0:
                lags = int(ceil(len(self.values) / 2.0))
            rho = self.autocorrelation(lags)
            for elem in rho[1:]:
                if -limit <= elem <= limit:
                    continue
                else:
                    outliers += 1

            if float(outliers) / (len(rho) - 1) >= 0.05:
                return False
            return True
        elif method == 'ljung-box':
            # use 10 lags as proposed by R. Hyndman
            # last entry from p-values array
            pval = acorr_ljungbox(self.values, 10)[1][-1]
            return pval >= 0.05
        else:
            raise ValueError("Method %s is not a valid argument" % method)
