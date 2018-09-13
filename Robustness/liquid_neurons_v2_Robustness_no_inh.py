import pyNN.brian as sim
from pyNN.random import RandomDistribution, NumpyRNG
import numpy as np
import matplotlib.pyplot as plt

N_e = 100 # number of excitatory neurons
Exc_in = 32 # number of excitatory inputs
Inh_in = 32 # number of inhibitatory inputs
weight_inp = [0.08, 0.12, 0.16, 0.20, 0.24, 0.28] # define the initial value of the input signal weight 

run_time = 500 * 500 # define the simulation time per run unit ms
exp_des = 'Robustness_no_inh' # define the exp no for data storage



rate = 15 # unit Hz
spike_times = RandomDistribution('uniform', (0.00, run_time)).next(int((run_time - 0.00) * rate * 1e-3))
# ==========simulation setup=====================


sim.setup()

# ==========generate OR read in the input spikes data=====================
Excinp = sim.Population(Exc_in, sim.SpikeSourceArray(spike_times = spike_times))
Inhinp = sim.Population(Inh_in, sim.SpikeSourceArray(spike_times = spike_times))


# ==========create neuron population=====================
# todo: the initail parameters of neurons might be modified
cell_type_parameters = {'tau_refrac': 0.1, 'v_thresh': -50.0, 'tau_m': 20.0, 'tau_syn_E': 0.5, 'v_rest': -65.0,\
						'cm': 1.0, 'v_reset': -65.0, 'tau_syn_I': 0.5, 'i_offset': 0.0}
# print(sim.IF_curr_alpha.default_parameters)

cell_type = sim.IF_cond_exp(**cell_type_parameters) # neuron type of population
Pexc = sim.Population(N_e, cell_type, label = "excitotary neurons") # excitatory neuron population

# todo: the Population structure

# ==========create neuron connection======================
# the network topology is based on 'compensating inhomogeneities of neuromorphic VLSI devices via short-term synaptic plasticity'

# todo: generate a concrete connection probability and weight (not chaged for each run)

# connection probability
# todo: connection probability based on position
P_c = 0.1 # scaling factor of connection probability; make sure the largest p (i.e. p_ii if (N_e > N_i)) is smaller than 1
p_ee = P_c


connector_ee = sim.FixedProbabilityConnector(p_connect = p_ee) # connector algorithm 



# connection weight
# todo: weight distribution according to position
W_c = 0.5 # scaling factor of weight
w_ee = W_c * 1


# type of synaptic connection
# syn = sim.StaticSynapse(weight = 0.05, delay = 0.5)
depressing_synapse_ee = sim.TsodyksMarkramSynapse(weight = w_ee, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)



# connect the neuronal network
E_E_connections = sim.Projection(Pexc, Pexc, connector_ee, depressing_synapse_ee, receptor_type = 'excitatory', label = "excitatory to excitatory") # excitatory to excitatory connection


# ==========injecting neuron currents OR connect to the input signal=======================

syn = sim.StaticSynapse(weight = weight_inp[0], delay = 0.5)
Ie_E_connections = sim.Projection(Excinp, Pexc, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'excitatory', label = "excitatory input to excitatory neurons")
Ii_E_connections = sim.Projection(Inhinp, Pexc, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'inhibitory', label = "inhibitory input to excitatory neurons")

# ==========define the parameter need to be record=================
Pexc.record('spikes')

for i, j in enumerate(weight_inp):
	Ie_E_connections.set(weight = j)
	Ii_E_connections.set(weight = j)
	sim.run(run_time)
	# ==========retrieve the data======================
	spikes = Pexc.get_data()
	print ('current sim %d, total %d' %(i + 1, len(weight_inp)))
	sim.reset()
	E_E_connections.set(weight = w_ee, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)

# =========simulation end==============
sim.end()

ave_fir = np.array([])

for i, j in enumerate(weight_inp):
	sum_spiketrain = 0
	for spiketrain in spikes.segments[i].spiketrains:
		sum_spiketrain = sum_spiketrain + np.size(spiketrain)
	average_fir = sum_spiketrain / (N_e * run_time * 1e-3) # unit s^-1 (i.e. Hz)
	if i == 0:
		ave_fir = np.hstack((ave_fir, np.array([j, average_fir])))
	else:
		ave_fir = np.vstack((ave_fir, np.array([j, average_fir])))

# print ave_fir
np.savetxt('./Robustness/robustness_%s' %exp_des, ave_fir, delimiter = ',')
