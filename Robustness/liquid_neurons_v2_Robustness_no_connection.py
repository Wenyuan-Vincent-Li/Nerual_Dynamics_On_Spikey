import pyNN.brian as sim
from pyNN.random import RandomDistribution, NumpyRNG
import numpy as np
import matplotlib.pyplot as plt

N_e = 75 # number of excitatory neurons
N_i = 25 # number of inhibitatory neurons
Exc_in = 32 # number of excitatory inputs
Inh_in = 32 # number of inhibitatory inputs
weight_inp = [0.08, 0.12, 0.16, 0.20, 0.24, 0.28] # define the initial value of the input signal weight 

run_time = 500 * 500 # define the simulation time per run unit ms
exp_des = 'Robustness_no_connection' # define the exp no for data storage



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
Pinh = sim.Population(N_i, cell_type, label = "inhibitatory neurons") # inhibitoty neuron population
all_cells = sim.Assembly(Pexc, Pinh) # assembly for all neuron population for the purpose of data recording
# todo: the Population structure


# ==========injecting neuron currents OR connect to the input signal=======================

syn = sim.StaticSynapse(weight = weight_inp[0], delay = 0.5)
Ie_E_connections = sim.Projection(Excinp, Pexc, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'excitatory', label = "excitatory input to excitatory neurons")
Ii_E_connections = sim.Projection(Inhinp, Pexc, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'inhibitory', label = "inhibitory input to excitatory neurons")
Ie_I_connections = sim.Projection(Excinp, Pinh, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'excitatory', label = "excitatory input to inhibitory neurons")
Ii_I_connections = sim.Projection(Inhinp, Pinh, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'inhibitory', label = "inhibitory input to inhibitory neurons")

# ==========define the parameter need to be record=================
all_cells.record('spikes')
# Excinp.record('spikes')

for i, j in enumerate(weight_inp):
	Ie_E_connections.set(weight = j)
	Ii_E_connections.set(weight = j)
	Ie_I_connections.set(weight = j)
	Ii_E_connections.set(weight = j)
	sim.run(run_time)
	# ==========retrieve the data======================
	spikes = all_cells.get_data()
	print ('current sim %d, total %d' %(i + 1, len(weight_inp)))
	sim.reset()

# =========simulation end==============
sim.end()

ave_fir = np.array([])

for i, j in enumerate(weight_inp):
	sum_spiketrain = 0
	for spiketrain in spikes.segments[i].spiketrains:
		sum_spiketrain = sum_spiketrain + np.size(spiketrain)
	average_fir = sum_spiketrain / ((N_e + N_i) * run_time * 1e-3) # unit s^-1 (i.e. Hz)
	if i == 0:
		ave_fir = np.hstack((ave_fir, np.array([j, average_fir])))
	else:
		ave_fir = np.vstack((ave_fir, np.array([j, average_fir])))

# print ave_fir
np.savetxt('./Robustness/robustness_%s' %exp_des, ave_fir, delimiter = ',')


# # =========plot figure=============
# fig_settings = {'lines.linewidth': 0.5, 'axes.linewidth': 0.5, 'axes.labelsize': 'small', 'legend.fontsize': 'small', 'font.size': 8} # define figure settings
# plt.rcParams.update(fig_settings) # update figure settings
# fig = plt.figure(1, figsize = (5, 5)) # create a figure

# def plot_spiketrains(segment):
# 	for spiketrain in segment.spiketrains:
# 		y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
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
# plot_spiketrains(spikes.segments[1])
# # plt.subplot(4, 1, 3)
# # plot_signal(spikes.segments[0].analogsignalarrays[0], 0)
# # plt.subplot(3, 1, 3)
# # plot_spiketrains(spikes.segments[2])

# # plt.xlabel("time (%s)" % spike.segments[0].analogsignalarrays[0].times.units._dimensionality.string)
# plt.setp(plt.gca().get_xticklabels(), visible = True)

# # todo: check the structure of segment
# plt.show()