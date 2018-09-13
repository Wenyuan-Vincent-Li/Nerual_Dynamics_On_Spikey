from pyNN.random import RandomDistribution, NumpyRNG
import numpy as np 

# first period signal
start_time_1 = 0.00
stop_time_1 = 150.00
rate_1 = 100 # unit Hz

# second period signal
start_time_2 = stop_time_1
stop_time_2 = 350
rate_2 = 200

# third period signal
start_time_3 = stop_time_2
stop_time_3 = 500.00
rate_3 = 100 # unit Hz


spike_times_1 = RandomDistribution('uniform', (start_time_1, stop_time_1), rng = NumpyRNG(seed = 72386)).next(int((stop_time_1 - start_time_1) * rate_1 * 1e-3))

spike_times_2 = RandomDistribution('uniform', (start_time_2, stop_time_2), rng = NumpyRNG(seed = 72389)).next(int((stop_time_2 - start_time_2) * rate_2 * 1e-3))

spike_times_3 = RandomDistribution('uniform', (start_time_3, stop_time_3), rng = NumpyRNG(seed = 72389)).next(int((stop_time_3 - start_time_3) * rate_3 * 1e-3))

print type(spike_times_1)
spike_time = np.concatenate((spike_times_1, spike_times_2, spike_times_3), axis = 0)

print np.ndim(spike_time)