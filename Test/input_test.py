import pyNN.brian as sim
from pyNN.random import RandomDistribution, NumpyRNG

Exc_in = 32
Inh_in = 32
noSpikes = 20 # number of spikes per chanel per simulation run
stimSpikes = RandomDistribution('uniform', low = 0, high = 500.0, rng = NumpyRNG(seed = 72386)).next([Exc_in + Inh_in, noSpikes]) # generate a time uniform distributed signal with Exc_in + Inh_in chanels and noSpikes for each chanel
# print stimSpikes

for i in range(Exc_in):
	if i == 0:
		Excinp = sim.Population(1, sim.SpikeSourceArray(spike_times = stimSpikes[i,:]))
	else:
		spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times = stimSpikes[i,:]))
		Excinp = Excinp + spike_source

for i in range(Inh_in):
	if i ==0:
		Inhinp = sim.Population(1, sim.SpikeSourceArray(spike_times = stimSpikes[i + Exc_in,:]))
	else:
		spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times = stimSpikes[i + Exc_in,:]))
		Inhinp = Inhinp + spike_source

# for p in Excinp.populations:
# 	print ("%-23s %4d %s" %(p.label, p.size, p.celltype.__class__.__name__))