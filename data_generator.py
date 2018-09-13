import logging
import random as rm
import numpy as np 
lg = logging.getLogger('data_gen')
lg.setLevel(logging.INFO)

filename = 'testingdata'
epochtime = 500
datasize = 100
source_rate = [40, 5]

label = [rm.randint(0, 1) for i in range(datasize)]
spike_source_0 = spike_source_1 = []
for i in range(len(label)):
	spike_time_0 = [rm.uniform(i * epochtime, (i + 1) * epochtime) for j in range(int(source_rate[label[i]] * epochtime * 1e-3))]
	spike_time_1 = [rm.uniform(i * epochtime, (i + 1) * epochtime) for j in range(int((source_rate[1 - label[i]]) * epochtime * 1e-3))]
	spike_source_0 = np.hstack((spike_source_0, spike_time_0))
	spike_source_1 = np.hstack((spike_source_1, spike_time_1))
np.savetxt(filename + '_spikesource_0.txt', spike_source_0, delimiter = ',')
np.savetxt(filename +'_spikesource_1.txt', spike_source_1, delimiter = ',')
np.savetxt(filename + '_label.txt', label, delimiter = ',')