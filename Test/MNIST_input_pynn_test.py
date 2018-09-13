import pyNN.brian as sim
import numpy as np 
training_data = np.loadtxt('training_data_0_1.txt', delimiter = ',')
training_label = training_data[:, -1]
training_rate = training_data[:, 0 : 64]
# print training_rate[1, :]
inputpop = []
sim.setup()
for i in range(np.size(training_rate, 1)):
	inputpop.append(sim.Population(1, sim.SpikeSourcePoisson(rate = abs(training_rate[0, i]))))

# print inputpop[0].get('rate')
# inputpop[0].set(rate = 8)
# print inputpop[0].get('rate')

pop = sim.Population(1, sim.IF_cond_exp(), label = 'exc')

prj1 = sim.Projection(inputpop[0], pop, sim.OneToOneConnector(), synapse_type = sim.StaticSynapse(weight = 0.04, delay = 0.5), receptor_type = 'inhibitory')
print prj1.get('weight', format = 'list')