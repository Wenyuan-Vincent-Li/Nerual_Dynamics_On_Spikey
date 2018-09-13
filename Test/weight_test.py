import pyNN.brian as sim
from pyNN.random import RandomDistribution, NumpyRNG
import numpy as np
import matplotlib.pyplot as plt

ite_no = 3

sim.setup()

# ==========create neuron population=====================
# todo: the initail parameters of neurons might be modified
cell_type_parameters = {'tau_refrac': 0.1, 'v_thresh': -50.0, 'tau_m': 20.0, 'tau_syn_E': 0.5, 'v_rest': -65.0,\
						'cm': 1.0, 'v_reset': -65.0, 'tau_syn_I': 0.5, 'i_offset': 0.0}
# print(sim.IF_curr_alpha.default_parameters)

cell_type = sim.IF_cond_exp(**cell_type_parameters) # neuron type of population
Pexc = sim.Population(3, cell_type, label = "excitotary neurons") # excitatory neuron population

# ==========generate OR read in the input spikes data=====================
noSpikes = 20 # number of spikes per chanel per simulation run
stimSpikes = RandomDistribution('uniform', low = 0, high = 500, rng = NumpyRNG(seed = 72386)).next(noSpikes) # generate a time uniform distributed signal with Exc_in + Inh_in chanels and noSpikes for each chanel

Excinp = sim.Population(3, sim.SpikeSourceArray(spike_times = stimSpikes))

syn = sim.StaticSynapse(weight = 0.05, delay = 0.5)
Ie_A_connections = sim.Projection(Excinp, Pexc, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'excitatory', label = "excitatory input")

for i in range(ite_no):
	sim.run(100)
	# ==========write the data======================
	sim.reset()
	Ie_A_connections.set(weight = (i + 2) * 0.05)
	print Ie_A_connections.get('weight', format = 'list')

# Ie_A_connections.set(weight = 0.02)

# print Ie_A_connections.get('weight', format = 'list')

# sim.run(200)

# sim.reset()
# Ie_A_connections.set(weight = 0.03)
# print Ie_A_connections.get('weight', format = 'list')

# sim.run(100)

sim.end()