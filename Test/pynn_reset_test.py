import pyNN.brian as sim
from pyNN.random import RandomDistribution, NumpyRNG
import numpy as np 
import matplotlib.pyplot as plt

# np.seterr(all='raise')


def plot_spiketrains(segment):
		for spiketrain in segment.spiketrains:
			y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
			# print spiketrain
			plt.plot(spiketrain, y, '.')
			plt.ylabel(segment.name)
			plt.setp(plt.gca().get_xticklabels(), visible = False)

def plot_signal(signal, index, colour = 'b'):
	label = "Neuron %d " % index
	plt.plot(signal.times, signal, colour, label = label)
	plt.ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))
	plt.setp(plt.gca().get_xticklabels(), visible = False)



run_time = 500
no_run = 3

sim.setup()
# Excinp = sim.Population(10, sim.SpikeSourcePoisson(rate = 20.0, start = 0, duration = run_time))
stimSpikes = RandomDistribution('uniform', low = 0, high = 500.0).next([1, 10])
Excinp = sim.Population(10, sim.SpikeSourceArray(spike_times = stimSpikes[0, :]))
# cell_type_parameters = {'tau_refrac': 0.1, 'v_thresh': -50.0, 'tau_m': 20.0, 'tau_syn_E': 0.5, 'v_rest': -65.0,\
						# 'cm': 1.0, 'v_reset': -65.0, 'tau_syn_I': 0.5, 'i_offset': 0.0}
# print(sim.IF_curr_alpha.default_parameters)

# cell_type = sim.IF_cond_exp(**cell_type_parameters) # neuron type of population
Pexc = sim.Population(10, sim.EIF_cond_exp_isfa_ista(), label = "excitotary neurons")
# Pexc.set(tau_refrac = 0.1, v_thresh = -50.0, tau_m = 20.0, tau_syn_E = 0.5, v_rest = -65.0, \
# 		cm = 1.0, v_reset = -65, tau_syn_I = 0.5, i_offset = 0.0)
# Pexc.initialize(**cell_type_parameters)
# print Pexc.celltype.default_initial_values
# print Pexc.get('tau_m')
# syn = sim.StaticSynapse(weight = 0.05, delay = 0.5)
depressing_synapse_ee = sim.TsodyksMarkramSynapse(weight = 0.05, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
facilitating_synapse_ee = sim.TsodyksMarkramSynapse(weight = 0.05, delay = 0.5, U = 0.04, tau_rec = 100.0, tau_facil = 1000)
static_synapse = sim.StaticSynapse(weight = 0.05, delay = 0.5)

Input_E_connection = sim.Projection(Excinp, Pexc, sim.AllToAllConnector(), static_synapse, receptor_type = 'excitatory')


E_E_connection = sim.Projection(Pexc, Pexc, sim.FixedProbabilityConnector(p_connect = 0.5), depressing_synapse_ee, receptor_type = 'excitatory')

Excinp.record('spikes')
# Excinp[1].record('v')
Pexc.record('spikes')
Pexc[5:6].record('v')

for i in range(no_run):
	sim.run(run_time)
	spikes = Excinp.get_data()
	spike = Pexc.get_data()
	# print connection.get('weight',format = 'array')
	# print E_E_connection.get('weight', format = 'array')
	# print connection.get('tau_facil', format = 'array')
	# connection.set(weight = 0.05, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
	# connection.set(weight = 0.05, delay = 0.5, U = 0.05, tau_rec = 100.0, tau_facil = 1000)
	# E_E_connection.set(weight = 0.05)
	# print Pexc.get('i_offset')
	# Pexc.set(tau_refrac = 100, v_thresh = -50.0, tau_m = 20.0, tau_syn_E = 0.5, v_rest = -65.0, \
	# 		cm = 1.0, v_reset = -65000.0, tau_syn_I = 0.5, i_offset = 0.0)
	# print Pexc.get('v_rest','cm', 'v_reset', 'tau_syn_I', 'i_offset')
	# print spikes.segments[i].spiketrains[2]
	# print type(spike.segments[i])
	fig_settings = {'lines.linewidth': 0.5, 'axes.linewidth': 0.5, 'axes.labelsize': 'small', 'legend.fontsize': 'small', 'font.size': 8} # define figure settings
	plt.rcParams.update(fig_settings) # update figure settings
	fig = plt.figure(1, figsize = (5, 5)) # create a figure

	# todo: figure out the meaning of each command
	plt.subplot(2, 1, 1)
	plot_spiketrains(spikes.segments[i])
	plt.subplot(2, 1, 2)
	plot_spiketrains(spike.segments[i])
	# plt.subplot(4, 1, 3)
	# plot_signal(spikes.segments[0].analogsignalarrays[0], 0)
	# plt.subplot(3, 1, 3)
	# plot_signal(spike.segments[1].analogsignalarrays[0], 0)

	# plt.xlabel("time (%s)" % spike.segments[0].analogsignalarrays[0].times.units._dimensionality.string)
	plt.setp(plt.gca().get_xticklabels(), visible = True)

	# todo: check the structure of segment

	plt.show()

	sim.reset()
	# connection.set(weight = 0.05, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
	Input_E_connection.set(weight = 0.05, delay = 0.5)
	# E_E_connection.set(weight = 0.05, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
	E_E_connection.set(weight = 0.05, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
	print(Excinp.get('spike_times'))
	stimSpikes = RandomDistribution('uniform', low = 0, high = 500.0).next([1, 10])
	Excinp.set(spike_times = stimSpikes[0, :])


# spikes = Excinp.get_data()
# spike = Pexc.get_data()

sim.end()



# print len(spike.segments)
# print spike.segments[0].spiketrains[2]
# print spikes.segments[0].spiketrains[2]


# for i in range(len(spike.segments)):
# 	rastor_matrix = np.empty(shape = [0, 2])
# 	for spiketrain in spike.segments[i].spiketrains:
# 		y = np.array(np.ones_like(spiketrain) * spiketrain.annotations['source_id'])
# 		spike_time = np.array(spiketrain)
# 		neuron_spike = np.column_stack((spike_time, y))
# 		rastor_matrix = np.vstack((rastor_matrix, neuron_spike))
# 		# print type(np.array(spiketrain)), type(y)
# 		# print np.size (np.array(spiketrain))
# 	np.savetxt('exp=%d.txt' %i, rastor_matrix)




# Pexc.write_data("mem_test_robostness.mat")
# sim.reset()
# sim.run(500)
# spikes = Excinp.get_data()
# spike = Pexc.get_data()
# sim.end()
# Pexc.write_data('testspikes.mat')

# print len(spikes.segments)
# print spikes.segments[0].spiketrains[2]
# print spikes.segments[0].spiketrains[3]
# print spike.segments[0].spiketrains[2]
# print spike.segments[0].spiketrains[5]


# =========plot figure=============
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
# plot_spiketrains(spike.segments[0])
# # plt.subplot(4, 1, 3)
# # plot_signal(spikes.segments[0].analogsignalarrays[0], 0)
# # plt.subplot(3, 1, 3)
# # plot_signal(spike.segments[1].analogsignalarrays[0], 0)

# # plt.xlabel("time (%s)" % spike.segments[0].analogsignalarrays[0].times.units._dimensionality.string)
# plt.setp(plt.gca().get_xticklabels(), visible = True)

# # todo: check the structure of segment

# plt.show()