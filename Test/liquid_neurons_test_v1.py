import pyNN.brian as sim
from pyNN.random import RandomDistribution, NumpyRNG
import numpy as np
import matplotlib.pyplot as plt

N_e = 75 # number of excitatory neurons
N_i = 25 # number of inhibitatory neurons
Exc_in = 32 # number of excitatory inputs
Inh_in = 32 # number of inhibitatory inputs
# No_inp = {'exc': 32, 'inh': 32}

run_time = 500 # define the simulation time

# ==========generate OR read in the input spikes data=====================
noSpikes = 20 # number of spikes per chanel per simulation run
stimSpikes = RandomDistribution('uniform', low = 0, high = run_time, rng = NumpyRNG(seed = 72386)).next([Exc_in + Inh_in, noSpikes]) # generate a time uniform distributed signal with Exc_in + Inh_in chanels and noSpikes for each chanel
# todo: 64 chanel represents different data

sim.setup() # start buiding up the network topology

# ==========create the input signal neuron population==================
# form the Exc_in chanels excitatory inputs as a assembly Inhinp
for i in range(Exc_in):
	if i == 0:
		Excinp = sim.Population(1, sim.SpikeSourceArray(spike_times = stimSpikes[i,:]))
	else:
		spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times = stimSpikes[i,:]))
		Excinp = Excinp + spike_source


# form the Inh_in chanels excitatory inputs as a assembly Inhinp
for i in range(Inh_in):
	if i == 0:
		Inhinp = sim.Population(1, sim.SpikeSourceArray(spike_times = stimSpikes[i + Exc_in,:]))
	else:
		spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times = stimSpikes[i + Exc_in,:]))
		Inhinp = Inhinp + spike_source


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

# ==========create neuron connection======================
# the network topology is based on 'compensating inhomogeneities of neuromorphic VLSI devices via short-term synaptic plasticity'

# todo: generate a concrete connection probability and weight (not chaged for each run)

# connection probability
# todo: connection probability based on position
P_c = 0.2 # scaling factor of connection probability; make sure the largest p (i.e. p_ii if (N_e > N_i)) is smaller than 1
p_ee = P_c
p_ie = 2 * P_c
p_ei = N_e / N_i * P_c
p_ii = 2 * N_e / N_i * P_c

connector_ee = sim.FixedProbabilityConnector(p_connect = p_ee) # connector algorithm 
connector_ie = sim.FixedProbabilityConnector(p_connect = p_ie)
connector_ei = sim.FixedProbabilityConnector(p_connect = p_ei)
connector_ii = sim.FixedProbabilityConnector(p_connect = p_ii)


# connection weight
# todo: weight distribution according to position
W_c = 0.5 # scaling factor of weight
w_ee = W_c
w_ie = 0.5 * W_c
w_ei = W_c
w_ii = 0.5 * W_c

# type of synaptic connection
syn = sim.StaticSynapse(weight = 0.05, delay = 0.5)
depressing_synapse_ee = sim.TsodyksMarkramSynapse(weight = w_ee, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
depressing_synapse_ii = sim.TsodyksMarkramSynapse(weight = w_ii, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)

tau_rec = RandomDistribution('normal', [100.0, 10.0])
facilitating_synapse_ie = sim.TsodyksMarkramSynapse(weight = w_ie, delay = 0.5, U = 0.04, tau_rec = tau_rec)
facilitating_synapse_ei = sim.TsodyksMarkramSynapse(weight = w_ei, delay = 0.5, U = 0.04, tau_rec = tau_rec)


# connect the neuronal network
E_E_connections = sim.Projection(Pexc, Pexc, connector_ee, depressing_synapse_ee, receptor_type = 'excitatory', label = "excitatory to excitatory") # excitatory to excitatory connection
I_I_connections = sim.Projection(Pinh, Pinh, connector_ii, depressing_synapse_ii, receptor_type = 'inhibitory', label = "inhibitory to inhibitory") # inhibitory to inhibitory connection
E_I_connections = sim.Projection(Pexc, Pinh, connector_ie, facilitating_synapse_ie, receptor_type = 'excitatory', label = "excitatory to inhibitory") # from excitatory to inhibitory connection
I_E_connections = sim.Projection(Pinh, Pexc, connector_ei, facilitating_synapse_ei, receptor_type = 'inhibitory', label = "inhibitory to excitatory") # from inhibitory to excitatory


# ==========injecting neuron currents OR connect to the input signal=======================
# todo: instead of current source, build up input neuron population carring with input information
# current = sim.ACSource(start = 0.0, stop = 400.0, amplitude = 10.0, offset = 10.0, frequency = 100.0, phase = 0.0)
# current.inject_into(Pexc[0 : 9])
# todo: randomly choose the neurons to inject currents

# connect the input signal
# Ie_A_connections = sim.Projection(Excinp, all_cells, sim.FixedNumberPreConnector(5, with_replacement = True), syn, receptor_type = 'excitatory', label = "excitatory input")
# Ii_A_connections = sim.Projection(Inhinp, all_cells, sim.FixedNumberPreConnector(5, with_replacement = True), syn, receptor_type = 'inhibitory', label = "inhibitory input")

Ie_A_connections = sim.Projection(Excinp, all_cells, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'excitatory', label = "excitatory input")
Ii_A_connections = sim.Projection(Inhinp, all_cells, sim.FixedProbabilityConnector(p_connect = 0.5), syn, receptor_type = 'inhibitory', label = "inhibitory input")
# todo: static weight modification
# todo: FixedNumberPreConnector should be a distribution


# ==========define the parameter need to be record=================
all_cells.record('spikes')
Pexc[[5]].record('v')

# ==========start the simulation====================
sim.run(run_time)

# ==========retrieve the data======================
spikes = all_cells.get_data()
mem_plot = Pexc.get_data()

all_cells.write_data("spikes_test.mat")
Pexc.write_data("mem_test.mat")
# =========simulation end==============
sim.end()

# =========plot figure=============
fig_settings = {'lines.linewidth': 0.5, 'axes.linewidth': 0.5, 'axes.labelsize': 'small', 'legend.fontsize': 'small', 'font.size': 8} # define figure settings
plt.rcParams.update(fig_settings) # update figure settings
fig = plt.figure(1, figsize = (6, 8)) # create a figure

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

# todo: figure out the meaning of each command
plt.subplot(2, 1, 1)
plot_spiketrains(spikes.segments[0])
plt.subplot(2, 1, 2)
plot_signal(mem_plot.segments[0].analogsignalarrays[0], 15)
plt.xlabel("time (%s)" % mem_plot.segments[0].analogsignalarrays[0].times.units._dimensionality.string)
plt.setp(plt.gca().get_xticklabels(), visible = True)

# todo: check the structure of segment

plt.show()
fig.savefig('no_explosion.png')