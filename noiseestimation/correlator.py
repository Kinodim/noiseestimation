from math import sqrt
import numpy as np

class Correlator:
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
        for elem in rho:
            if -limit <= elem <= limit:
                continue
            else:
                outliers += 1

        if outliers / len(rho) >= 0.05:
            return False
        return True
