import pyNN.brian as sim
from pyNN.random import RandomDistribution, NumpyRNG
import numpy as np
import matplotlib.pyplot as plt

import pyNN.brian as sim
from pyNN.random import RandomDistribution, NumpyRNG
import numpy as np
import matplotlib.pyplot as plt

N_e = 75 # number of excitatory neurons
N_i = 25 # number of inhibitatory neurons
Exc_in = 5 # number of excitatory inputs
Inh_in = 32 # number of inhibitatory inputs
weight_inp = 0.14 # define the initial value of the input signal weight 

run_time = 50 # define the simulation time per run unit ms
no_run = 2 # how many runs you will have

# ==========define the input signal rates=====================
# first period signal
start_time_1 = 0.00
stop_time_1 = 150.00
rate_1 = 50 # unit Hz

# second period signal
start_time_2 = stop_time_1
stop_time_2 = 350
rate_2 = 20

# third period signal
start_time_3 = stop_time_2
stop_time_3 = 500.00
rate_3 = 10 # unit Hz



spike_time_run = []

for i in range(no_run):
	spike_times_1 = RandomDistribution('uniform', (run_time * i + start_time_1, run_time * i + stop_time_1)).next(int((stop_time_1 - start_time_1) * rate_1 * 1e-3))
	spike_times_2 = RandomDistribution('uniform', (run_time * i + start_time_2, run_time * i + stop_time_2)).next(int((stop_time_2 - start_time_2) * rate_2 * 1e-3))
	spike_times_3 = RandomDistribution('uniform', (run_time * i + start_time_3, run_time * i + stop_time_3)).next(int((stop_time_3 - start_time_3) * rate_3 * 1e-3))
	# spike_times_4 = RandomDistribution('uniform', (run_time * i + start_time_4, run_time * i + stop_time_4)).next(int((stop_time_3 - start_time_3) * rate_3 * 1e-3))
	spike_time = np.concatenate((spike_times_1, spike_times_2, spike_times_3), axis = 0)
	# print spike_time
	# print type(spike_times_1)
	spike_time_run = np.hstack((spike_time_run, spike_time))

# print spike_time_run
# ==========simulation setup=====================


sim.setup()

# ==========generate OR read in the input spikes data=====================
Excinp = sim.Population(Exc_in, sim.SpikeSourceArray(spike_times = spike_time_run))
Inhinp = sim.Population(Inh_in, sim.SpikeSourceArray(spike_times = spike_time_run))


# # ==========create neuron population=====================
# # todo: the initail parameters of neurons might be modified
# cell_type_parameters = {'tau_refrac': 0.1, 'v_thresh': -50.0, 'tau_m': 20.0, 'tau_syn_E': 0.5, 'v_rest': -65.0,\
# 						'cm': 1.0, 'v_reset': -65.0, 'tau_syn_I': 0.5, 'i_offset': 0.0}
# # print(sim.IF_curr_alpha.default_parameters)

# cell_type = sim.IF_cond_exp(**cell_type_parameters) # neuron type of population
# Pexc = sim.Population(N_e, cell_type, label = "excitotary neurons") # excitatory neuron population
# Pinh = sim.Population(N_i, cell_type, label = "inhibitatory neurons") # inhibitoty neuron population
# all_cells = sim.Assembly(Pexc, Pinh) # assembly for all neuron population for the purpose of data recording
# # todo: the Population structure

# # ==========create neuron connection======================
# # the network topology is based on 'compensating inhomogeneities of neuromorphic VLSI devices via short-term synaptic plasticity'

# # todo: generate a concrete connection probability and weight (not chaged for each run)

# # connection probability
# # todo: connection probability based on position
# P_c = 0.1 # scaling factor of connection probability; make sure the largest p (i.e. p_ii if (N_e > N_i)) is smaller than 1
# p_ee = P_c
# p_ie = 2 * P_c
# p_ei = N_e / N_i * P_c
# p_ii = 2 * N_e / N_i * P_c

# connector_ee = sim.FixedProbabilityConnector(p_connect = p_ee) # connector algorithm 
# connector_ie = sim.FixedProbabilityConnector(p_connect = p_ie)
# connector_ei = sim.FixedProbabilityConnector(p_connect = p_ei)
# connector_ii = sim.FixedProbabilityConnector(p_connect = p_ii)


# # connection weight
# # todo: weight distribution according to position
# W_c = 0.5 # scaling factor of weight
# w_ee = W_c * 1
# w_ie = 0.5 * W_c * 0.1
# w_ei = W_c * 3
# w_ii = 0.5 * W_c * 0.5

# # type of synaptic connection
# # syn = sim.StaticSynapse(weight = 0.05, delay = 0.5)
# depressing_synapse_ee = sim.TsodyksMarkramSynapse(weight = w_ee, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
# depressing_synapse_ii = sim.TsodyksMarkramSynapse(weight = w_ii, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)

# # tau_rec = RandomDistribution('normal', [100.0, 10.0])
# facilitating_synapse_ie = sim.TsodyksMarkramSynapse(weight = w_ie, delay = 0.5, U = 0.04, tau_rec = 100.0, tau_facil = 1000)
# facilitating_synapse_ei = sim.TsodyksMarkramSynapse(weight = w_ei, delay = 0.5, U = 0.04, tau_rec = 100.0, tau_facil = 1000)



# # connect the neuronal network
# E_E_connections = sim.Projection(Pexc, Pexc, connector_ee, depressing_synapse_ee, receptor_type = 'excitatory', label = "excitatory to excitatory") # excitatory to excitatory connection
# I_I_connections = sim.Projection(Pinh, Pinh, connector_ii, depressing_synapse_ii, receptor_type = 'inhibitory', label = "inhibitory to inhibitory") # inhibitory to inhibitory connection
# E_I_connections = sim.Projection(Pexc, Pinh, connector_ie, facilitating_synapse_ie, receptor_type = 'excitatory', label = "excitatory to inhibitory") # from excitatory to inhibitory connection
# I_E_connections = sim.Projection(Pinh, Pexc, connector_ei, facilitating_synapse_ei, receptor_type = 'inhibitory', label = "inhibitory to excitatory") # from inhibitory to excitatory
# # print I_E_connections.get('tau_facil', format = 'array')
# # print E_E_connections.get('tau_facil', format = 'array')

# # ==========injecting neuron currents OR connect to the input signal=======================

# syn = sim.StaticSynapse(weight = weight_inp, delay = 0.5)
# Ie_E_connections = sim.Projection(Excinp, Pexc, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'excitatory', label = "excitatory input to excitatory neurons")
# Ii_E_connections = sim.Projection(Inhinp, Pexc, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'inhibitory', label = "inhibitory input to excitatory neurons")
# Ie_I_connections = sim.Projection(Excinp, Pinh, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'excitatory', label = "excitatory input to inhibitory neurons")
# Ii_I_connections = sim.Projection(Inhinp, Pinh, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'inhibitory', label = "inhibitory input to inhibitory neurons")

# # ==========define the parameter need to be record=================
# all_cells.record('spikes')
Excinp.record('spikes')

for i in range(no_run):
	sim.run_until(run_time * (i + 1))
	# ==========retrieve the data======================
	# spikes = all_cells.get_data()
	inp_spikes = Excinp.get_data()
	print ('current time %.1f, total %.1f' %(run_time * (i + 1), run_time * no_run))
	# print inp_spikes.segments[0].spiketrains
	sum_spiketrain = []
	for spiketrain in inp_spikes.segments[0].spiketrains:
		print spiketrain
		sum_spiketrain.append(sum(spiketrain > (i * run_time)))
	sum_spiketrain.extend([4, 5, 6])
	pre_spike_sort = np.argsort(sum_spiketrain)
	print sum_spiketrain, pre_spike_sort[::-1]
	
	# average_fir_out1 = sum_spiketrain / (10 * runtime * 1e-3) # unit Hz
	
	# fig_settings = {'lines.linewidth': 0.5, 'axes.linewidth': 0.5, 'axes.labelsize': 'small', 'legend.fontsize': 'small', 'font.size': 8} # define figure settings
	# plt.rcParams.update(fig_settings) # update figure settings
	# fig = plt.figure((i + 1), figsize = (5, 5)) # create a figure

	# def plot_spiketrains(segment):
	# 	for spiketrain in segment.spiketrains:
	# 		y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
	# 		print y
	# 		# print spiketrain
	# 		plt.plot(spiketrain, y, '.')
	# 		plt.ylabel(segment.name)
	# 		plt.setp(plt.gca().get_xticklabels(), visible = False)

	# def plot_signal(signal, index, colour = 'b'):
	# 	label = "Neuron %d " % index
	# 	plt.plot(signal.times, signal, colour, label = label)
	# 	plt.ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))
	# 	plt.setp(plt.gca().get_xticklabels(), visible = False)

	# # todo: figure out the meaning of each command
	# plt.subplot(2, 1, 1)
	# plot_spiketrains(spikes.segments[0])
	# plt.subplot(2, 1, 2)
	# plot_spiketrains(inp_spikes.segments[0])
	# # plt.subplot(4, 1, 3)
	# # plot_signal(spikes.segments[0].analogsignalarrays[0], 0)
	# # plt.subplot(3, 1, 3)
	# # plot_spiketrains(spikes.segments[2])

	# # plt.xlabel("time (%s)" % spike.segments[0].analogsignalarrays[0].times.units._dimensionality.string)
	# plt.setp(plt.gca().get_xticklabels(), visible = True)

	# # todo: check the structure of segment
	# plt.show()

	# print type(spikes.segments[1]), type(inp_spikes)

# spikes = all_cells.get_data()
# inp_spikes = Excinp.get_data()

# =========simulation end==============
sim.end()

cutoff = 5
cutoff -= 12
print cutoff
cutoff *= -1
print cutoff