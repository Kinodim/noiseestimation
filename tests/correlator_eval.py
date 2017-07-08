from numpy.random import randn
from noiseestimation.correlator import Correlator

num_runs = 50
arr_length = 1000

num_mehra_white = 0
num_ljungbox_white = 0
for i in range(num_runs):
    seq = randn(arr_length)
    cor = Correlator(seq)

    if cor.isWhite():
        num_ljungbox_white += 1
    if cor.isWhite('mehra'):
        num_mehra_white += 1

print("-" * 10)
print("%d / %d white (%.3f %%) using Mehra method" % (num_mehra_white, num_runs, 100 * float(num_mehra_white) / num_runs))
print("%d / %d white (%.3f %%) using Ljund-Box method" % (num_ljungbox_white, num_runs, 100 * float(num_ljungbox_white) / num_runs))
