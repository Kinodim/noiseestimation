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

