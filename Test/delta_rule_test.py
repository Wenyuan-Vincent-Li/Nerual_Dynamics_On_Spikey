import logging

import pyNN.brian as sim
import numpy as np 
import matplotlib.pyplot as plt

def change_predec_weights_learning(w1_1, w1_2, w2_1, w2_2, label, average_fir_out1, average_fir_out2):
	eta = 0.05

	if label == 1:
		if average_fir_out1 < 30:
			w1_1 = w1_1 + eta * w1_1
			w2_1 = w2_1 + eta * w2_1
			lg.info('change weights')
		if average_fir_out2 > 10:
			w1_2 = w1_2 - eta * w1_2
			w2_2 = w2_2 - eta * w2_2
			lg.info('change weights')
	else:
		if average_fir_out1 > 10:
			w1_1 = w1_1 - eta * w1_1
			w2_1 = w2_1 - eta * w2_1
			lg.info('change weights')
		if average_fir_out2 < 30:
			w1_2 = w1_2 + eta * w1_2
			w2_2 = w2_2 + eta * w2_2
			lg.info('change weights')
	return w1_1, w1_2, w2_1, w2_2


def learn_pattern(spikes, label, w1_1, w1_2, w2_1, w2_2):
	lg.info('performing learning')
	# determing winner population and class
	# print spikes.segments[0].spiketrains[0 : 10]
	# calculate the average firing rate
	sum_spiketrain = 0
	for spiketrain in spikes.segments[0].spiketrains[0 : 10]:
		sum_spiketrain = sum_spiketrain + np.size(spiketrain)
	average_fir_out1 = sum_spiketrain / (10 * runtime * 1e-3) # unit Hz
	sum_spiketrain = 0
	for spiketrain in Data_O.segments[0].spiketrains[10 : 20]:
		sum_spiketrain = sum_spiketrain + np.size(spiketrain)
	average_fir_out2 = sum_spiketrain / (10 * runtime * 1e-3) # unit Hz

	w1_1, w1_2, w2_1, w2_2 = change_predec_weights_learning(w1_1, w1_2, w2_1, w2_2, label, average_fir_out1, average_fir_out2)
	return w1_1, w1_2, w2_1, w2_2

logging.basicConfig()
lg = logging.getLogger('delta_rule_test')
lg.setLevel(logging.INFO)

training_label = np.loadtxt('training_example.txt', delimiter = ',', dtype = int)
# print training_label
connector1_1 = np.loadtxt('input1_output1_test.txt', delimiter = ',', dtype = bool)
connector1_2 = np.loadtxt('input1_output2_test.txt', delimiter = ',', dtype = bool)
connector2_1 = np.loadtxt('input2_output1_test.txt', delimiter = ',', dtype = bool)
connector2_2 = np.loadtxt('input2_output2_test.txt', delimiter = ',', dtype = bool)

runtime = 50
eta = 0.5 # learning rate
iter_no = 5 # learning iteration
source_rate = [20, 40]

w1_1 = 0.1
w1_2 = 0.1
w2_1 = 0.1
w2_2 = 0.1

# set up the classifier network
for i in range(len(training_label)):
# for i in range(1):
	lg.info('iteration number %d' %i)
	sim.setup()
	In_1 = sim.Population(10, sim.SpikeSourcePoisson(rate = source_rate[training_label[i]]))
	In_2 = sim.Population(10, sim.SpikeSourcePoisson(rate = source_rate[1 - training_label[i]]))
	In = In_1 + In_2

	Out_1 = sim.Population(10, sim.IF_cond_exp())
	Out_2 = sim.Population(10, sim.IF_cond_exp())

	Out = Out_1 + Out_2

	syn_1_1 = sim.StaticSynapse(weight = w1_1, delay = 0.5)
	syn_1_2 = sim.StaticSynapse(weight = w1_2, delay = 0.5)
	syn_2_1 = sim.StaticSynapse(weight = w2_1, delay = 0.5)
	syn_2_2 = sim.StaticSynapse(weight = w2_2, delay = 0.5)
	prj_1_1 = sim.Projection(In_1, Out_1, sim.ArrayConnector(connector1_1), syn_1_1, receptor_type = 'excitatory')
	prj_1_2 = sim.Projection(In_1, Out_2, sim.ArrayConnector(connector1_2), syn_1_2, receptor_type = 'excitatory')
	prj_2_1 = sim.Projection(In_2, Out_1, sim.ArrayConnector(connector2_1), syn_2_1, receptor_type = 'excitatory')
	prj_2_2 = sim.Projection(In_2, Out_2, sim.ArrayConnector(connector2_2), syn_2_2, receptor_type = 'excitatory')
	# print prj_1_1.get('weight', format = 'array')
	# weights = prj_1_1.get('weight', format = 'array')
	# weights = weights + eta * weights
	# prj_1_1.set(weight = weights)
	# print prj_1_1.get('weight', format = 'array')
	w1_1 = prj_1_1.get('weight', format = 'array')
	w1_2 = prj_1_2.get('weight', format = 'array')
	w2_1 = prj_2_1.get('weight', format = 'array')
	w2_2 = prj_2_2.get('weight', format = 'array')

	In.record('spikes')
	Out.record('spikes')

	sim.run(runtime)
	Data_I = In.get_data()
	Data_O = Out.get_data()
	w1_1, w1_2, w2_1, w2_2 = learn_pattern(Data_O, training_label[i], w1_1, w1_2, w2_1, w2_2)
	# print w1_1, w1_2, w2_1, w2_2

# for i in range(iter_no):
# 	sim.run(runtime)
# 	Data_I = In.get_data()
# 	Data_O = Out.get_data()
# 	# calculate the average firing rate
# 	sum_spiketrain = 0
# 	for spiketrain in Data_O.segments[i].spiketrains[0 : 10]:
# 		sum_spiketrain = sum_spiketrain + np.size(spiketrain)
# 	average_fir_out1 = sum_spiketrain / (10 * runtime * 1e-3) # unit Hz
# 	# print average_fir_out1
# 	if average_fir_out1 <= 30:
# 		w1_1 = w1_1 + eta * w1_1
# 		w2_1 = w2_1 + eta * w2_1

# 	sum_spiketrain = 0
# 	for spiketrain in Data_O.segments[i].spiketrains[10 : 20]:
# 		sum_spiketrain = sum_spiketrain + np.size(spiketrain)
# 	average_fir_out2 = sum_spiketrain / (10 * runtime * 1e-3) # unit Hz
# 	# print average_fir_out2
# 	if average_fir_out1 >= 10:
# 		w1_2 = w1_2 - eta * w1_2
# 		w2_2 = w2_2 - eta * w2_2
# 	sim.reset()
# 	print i 
# 	prj_1_1.set(weight = w1_1)
# 	prj_1_2.set(weight = w1_2)
# 	prj_2_1.set(weight = w2_1)
# 	prj_2_2.set(weight = w2_2)

# print average_fir

sim.end()

print w1_1, w1_2, w2_1, w2_2

# # =========plot figure=============
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
plot_spiketrains(Data_I.segments[0])
plt.subplot(2, 1, 2)
plot_spiketrains(Data_O.segments[0])
plt.setp(plt.gca().get_xticklabels(), visible = True)

plt.show()
