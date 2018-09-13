from pyNN.random import RandomDistribution, NumpyRNG
spiketimes = RandomDistribution('uniform', low = 0, high = 500, rng = NumpyRNG(seed = 72386)).next([2, 3])
# spike = spiketimes.next([2, 3])
# print spike
print spiketimes