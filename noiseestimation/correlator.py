from math import sqrt
import numpy as np

class Correlator:
    """
    Provides functionality related to the autocorrelation coefficients
    of a sequence of values. More specifically, it calculates whether
    this sequence is some white noise using its estimated autocorrelation
    and the times they exceed the 95% confidence interval.
    Adapted from: On the Identification of Variances and Adaptive
    Kalman Filtering, Mehra 1970
    """
    def __init__(self, values, max_lag):
        self.values = values
        self.max_lag = max_lag

    def covariance(self):
        C = []
        for k in range(self.max_lag + 1):
            c = .0
            for i in range(k, len(self.values)):
                c += self.values[i] * self.values[i-k]
            c /= len(self.values)
            C.append(c)
        return C

    def autocorrelation(self):
        C = self.covariance()
        rho = [c / C[0] for c in C]
        return rho

    def isWhite(self):
        limit = 1.96 / sqrt(len(self.values))
        outliers = 0
        rho = self.autocorrelation()
        for elem in rho[1:]:
            if -limit <= elem <= limit:
                continue
            else:
                outliers += 1

        print float(outliers) / (len(rho) - 1)
        if float(outliers) / (len(rho) - 1) >= 0.05:
            return False
        return True
