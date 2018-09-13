"""
classes and functions to create the network
"""

import logging
from pyNN.random import RandomDistribution, NumpyRNG
lg = logging.getLogger('network')
lg.setLevel(logging.INFO)

import numpy as np

class LSMBrain(object):
	"""
	container class for all circuits we modeled in LSM
	"""
	def __init__(self, pynn, config, process, input_rate):
		self.pynn = pynn
		self.config = config
		self.input_rate = input_rate
		self.Input = InputNeuron(pynn, config, process, input_rate) # create input neurons
		self.Res = ReserviorNeuron(pynn, config) # create reservior neurons
		self.Dec = OutputNeuron(pynn, config) # create decision neurons		
		self.process = process # training or testing process


	def wire_Input_to_Res(self):
		"""
		Set up the connection from the input neurons to the reservior neruons
		"""
		writeflag = 0
		if self.config['network']['structure'] != 'no_inh':
			connector_mat_input_I = np.loadtxt('input_resI.txt', dtype = bool, delimiter = ',')
			try:
				w_I = np.loadtxt('input_resI_weight.txt', dtype = float, delimiter = ',')
			except IOError:
				w = RandomDistribution('normal_clipped_to_boundary', mu = 0.04, sigma = 0.01, low = 0.0005, high = 0.1)
				writeflag = 1 # generating new weights and need to be written to data
			Inputprj_I = []
			for i in range(len(self.input_rate)):
				if not writeflag:
					w = np.array(w_I[i, :], ndmin = 2)
				connector_I = self.pynn.ArrayConnector(np.array(connector_mat_input_I[i, :], ndmin = 2))
				syn_I = self.pynn.StaticSynapse(weight = w, delay = 0.5)
				if self.input_rate[i] >= 0:
					receptor = 'excitatory'
				else:
					receptor = 'inhibitory'
				prj = self.pynn.Projection(self.Input.input_source[i], self.Res.ResNeuron[1], connector_I, syn_I, receptor_type = receptor)
				Inputprj_I.append(prj)
				if i == 0:
					prj_I = prj.get('weight', format = 'array')
				else:
					prj_I = np.vstack((prj_I, prj.get('weight', format = 'array')))
			if writeflag:
				np.savetxt('input_resI_weight.txt', prj_I, delimiter = ',')	

		connector_mat_input_E = np.loadtxt('input_resE.txt', dtype = bool, delimiter = ',')	
		try:
			w_E = np.loadtxt('input_resE_weight.txt', dtype = float, delimiter = ',')
		except IOError:
			w = RandomDistribution('normal_clipped_to_boundary', mu = 0.04, sigma = 0.01, low = 0.0005, high = 0.1)
			writeflag = 1 # generating new weights and need to be written to data

		Inputprj_E = []
		for i in range(len(self.input_rate)):
			if not writeflag:
				w = np.array(w_E[i, :], ndmin = 2)
			connector_E = self.pynn.ArrayConnector(np.array(connector_mat_input_E[i, :], ndmin = 2))
			syn_E = self.pynn.StaticSynapse(weight = w, delay = 0.5)
			if self.input_rate[i] >= 0:
				receptor = 'excitatory'
			else:
				receptor = 'inhibitory'
			prj = self.pynn.Projection(self.Input.input_source[i], self.Res.ResNeuron[0], connector_E, syn_E, receptor_type = receptor)
			Inputprj_E.append(prj)
			if i == 0:
				prj_E = prj.get('weight', format = 'array')
			else:
				prj_E = np.vstack((prj_E, prj.get('weight', format = 'array')))
		if writeflag:
			np.savetxt('input_resE_weight.txt', prj_E, delimiter = ',')

		if self.config['network']['structure'] != 'no_inh':
			return Inputprj_I, Inputprj_E
		else:
			return Inputprj_E



	def wire_Res_to_Dec(self, item, w_last = None):
		"""
		Set up the connection from the reservior neurons to the decision layer.
		Return a projection list. order: 1. Decsion pop 2. resvoir_exc, reservoir_inh
		"""
		prj = []
		res = ['E', 'I']
		dec = [x for x in range(len(self.Dec.dec_pops))]
		for i in range(len(self.Dec.dec_pops)):
			for j in range(len(self.Res.ResNeuron)):
				if self.process == 'testing':
					w = np.loadtxt('res%s_output%d_weight_No.%d.txt'%(res[j], dec[i], w_last), dtype = float, delimiter = ',')					
				else:
					if w_last == None:
						try:
							w = np.loadtxt('res%s_output%d_weight_No._%d.txt'%(res[j], dec[i], item), dtype = float, delimiter = ',')
							# print 'load txt file successfully'
						except IOError:
							w = RandomDistribution('normal_clipped_to_boundary', mu = 0.01, sigma = 0.01, low = 0.0005, high = 0.5)
					else:
						if self.config['network']['structure'] != 'no_inh':
							w = w_last[i * 2 + j]
						else:
							w = w_last[i * 1 + j]
				syn = self.pynn.StaticSynapse(weight = w, delay = 0.1)
				connector = self.pynn.ArrayConnector(np.loadtxt('res%s_output%d.txt'%(res[j], dec[i]), dtype = bool, delimiter = ','))
				prj.append(self.pynn.Projection(self.Res.ResNeuron[j], self.Dec.dec_pops[i], connector, syn, receptor_type = 'excitatory'))
		return prj

class InputNeuron(object):
	"""
	create input spike source
	"""
	def __init__(self, pynn, config, process, input_rate):
		self.pynn = pynn 
		self.config = config
		self.process = process
		self.input_rate = input_rate

		# create input spike source
		self.input_source = []
		num_input_neuron = config['network'].as_int('num_input_neuron')
		for i in range(num_input_neuron):
			self.input_source.append(self._create_spike_source(1, abs(input_rate[i])))
			# self.input_source[i].record('spikes') # This is only for debuging. Comment this off for experiment trials


	def _create_spike_source(self, num_neuron, spikerate):
		"""
		create spike sources for DLSM. 
		Parameters:
		num_input: how many different input sources
		num_input_neuron: how many neurons in each input
		"""
		# spike_source = self.pynn.Population(num_neuron, self.pynn.SpikeSourcePoisson(rate = spikerate))
		duration = self.config['simulation'].as_float('epochtime')
		stimspike = RandomDistribution('uniform', low = 0, high = duration).next(int(duration * spikerate * 1e-3))
		spike_source = self.pynn.Population(num_neuron, self.pynn.SpikeSourceArray(spike_times = stimspike))
		return spike_source

	def get_spikecountmat(self):
		# get the spikes in reservior neurons
		spikecountmat = []
		for i in range(len(self.input_source)):
			spikecountmat.append(self.input_source[i].get_data())
		return spikecountmat

class ReserviorNeuron(object):
	"""
	create reservior neruons
	"""
	def __init__(self, pynn, config):
		self.pynn = pynn 
		self.config = config

		# create reservior neurons
		if config['network']['structure'] == 'no_inh':
			num_res = [config['network'].as_int('num_exc')]
		else:
			num_res = [config['network'].as_int('num_exc'), config['network'].as_int('num_inh')] # a list contains two elements: 1st the excitatory neurons 2nd the inhibitory neurons
		res_label = [config['network']['label_exc'], config['network']['label_inh']]
		self.ResNeuron = []
		for i in range(len(num_res)):
			self.ResNeuron.append(self._create_neuron_group(num_res[i], res_label[i]))
			self.ResNeuron[i].record('spikes')
		# wire reservior neruons
		if not self.config['network']['structure'] == 'no_connection':
			self.prj = self.wire_Resneuron(self.ResNeuron) # a list record projection in reservior neurons


	def get_spikecountmat(self):
		# get the spikes in reservior neurons
		spikecountmat = []
		for i in range(len(self.ResNeuron)):
			spikecountmat.append(self.ResNeuron[i].get_data())
		return spikecountmat


	def _create_neuron_group(self, num_neurons, res_label):
		"""
		create a group of identical neurons and return them as a list of pynn.Population.
		"""
		threshold = self.config['network'].as_float('v_th')
		v_th = RandomDistribution('normal_clipped_to_boundary', mu = threshold, sigma = 0.05, low = threshold - 5, high = threshold + 5, rng = NumpyRNG(seed=72386))
		return self.pynn.Population(num_neurons, self.pynn.IF_cond_exp(v_thresh = threshold), label = res_label)

	def wire_Resneuron(self, pop):
		"""
		create the connection in reservior neurons
		parameters:
		pop: population list, containing both excitatory and inhibitory neurons
		"""
		# first wire the recurrent loop
		prj = []
		for i in range(len(pop)):
			prj.append(self.wire_rec_loop(pop[i]))
		if self.config['network']['structure'] != 'no_inh':
			prj.extend(self.wire_bridge_connection(pop))
		return prj

	def wire_rec_loop(self, pop):
		"""
		create the recurrent loop for the reservoir neruons
		"""
		writeflag = 0
		if pop.label == 'excitatory neuron':
			try:
				w_ee = np.loadtxt('resE_resE_weight.txt', dtype = float, delimiter = ',')
			except IOError:
				w_ee = RandomDistribution('normal_clipped_to_boundary', mu = 0.5, sigma = 0.05, low = 0.0005, high = 1)
				writeflag = 1 # generating new weights and need to be written to data

			if self.config['network']['structure'] == 'static':
				synapse_ee = self.pynn.StaticSynapse(weight = w_ee, delay = 0.2)
			else:
				synapse_ee = self.pynn.TsodyksMarkramSynapse(weight = w_ee, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
			
			connector_ee = self.pynn.ArrayConnector(np.loadtxt('resE_resE.txt', dtype = bool, delimiter = ',')) # connector algorithm 
			connections = self.pynn.Projection(pop, pop, connector_ee, synapse_ee, receptor_type = 'excitatory', label = "excitatory to excitatory")
			if writeflag:
				np.savetxt('resE_resE_weight.txt', connections.get('weight', format = 'array'), delimiter = ',')
		elif pop.label == 'inhibitory neuron':
			try:
				w_ii = np.loadtxt('resI_resI_weight.txt', dtype = float, delimiter = ',')
			except IOError:
				w_ii = RandomDistribution('normal_clipped_to_boundary', mu = 0.5 * 0.5 * 0.5, sigma = 0.001, low = 0.005, high = 0.1)
				writeflag = 1
			if self.config['network']['structure'] == 'static':
				synapse_ii = self.pynn.StaticSynapse(weight = w_ii, delay = 0.2)
			else:
				synapse_ii = self.pynn.TsodyksMarkramSynapse(weight = w_ii, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
			connector_ii = self.pynn.ArrayConnector(np.loadtxt('resI_resI.txt', dtype = bool, delimiter = ','))
			connections = self.pynn.Projection(pop, pop, connector_ii, synapse_ii, receptor_type = 'inhibitory', label = "inhibitory to inhibitory") # inhibitory to inhibitory connection
			if writeflag:
				np.savetxt('resI_resI_weight.txt', connections.get('weight', format = 'array'), delimiter = ',')
		return connections

	def wire_bridge_connection(self, pop):
		"""
		create bridge connection between excitatory and inhibitory neurons
		"""
		writeflag = 0
		try:
			w_ie = np.loadtxt('resI_resE_weight.txt', dtype = float, delimiter = ',')
		except IOError:
			w_ie = RandomDistribution('normal_clipped_to_boundary', mu = 0.5 * 3, sigma = 0.01, low = 0.0005, high = 2)
			writeflag = 1
		try:
			w_ei = np.loadtxt('resE_resI_weight.txt', dtype = float, delimiter = ',')
		except IOError:
			w_ei = RandomDistribution('normal_clipped_to_boundary', mu = 0.5 * 0.5 * 0.1, sigma = 0.01, low = 0.0005, high = 0.1)
			writeflag = 1
		
		if self.config['network']['structure'] == 'static':
			synapse_ie = self.pynn.StaticSynapse(weight = w_ie, delay = 0.5)
			synapse_ei = self.pynn.StaticSynapse(weight = w_ei, delay = 0.5)
		else:
			synapse_ie = self.pynn.TsodyksMarkramSynapse(weight = w_ie, delay = 0.5, U = 0.04, tau_rec = 100.0, tau_facil = 1000)
			synapse_ei = self.pynn.TsodyksMarkramSynapse(weight = w_ei, delay = 0.5, U = 0.04, tau_rec = 100.0, tau_facil = 1000)
		
		connector_ei = self.pynn.ArrayConnector(np.loadtxt('resE_resI.txt', dtype = bool, delimiter = ','))
		connector_ie = self.pynn.ArrayConnector(np.loadtxt('resI_resE.txt', dtype = bool, delimiter = ','))
		if pop[0].label == 'excitatory neuron' and pop[1].label == 'inhibitory neuron':
			E_I_connections = self.pynn.Projection(pop[0], pop[1], connector_ei, synapse_ei, receptor_type = 'excitatory', label = "excitatory to inhibitory") # from excitatory to inhibitory connection
			I_E_connections = self.pynn.Projection(pop[1], pop[0], connector_ie, synapse_ie, receptor_type = 'inhibitory', label = "inhibitory to excitatory") # from inhibitory to excitatory

		elif pop[0].label == 'inhibitory neuron' and pop[1].label == 'excitatory neuron':
			E_I_connections = self.pynn.Projection(pop[1], pop[0], connector_ei, synapse_ei, receptor_type = 'excitatory', label = "excitatory to inhibitory") # from excitatory to inhibitory connection
			I_E_connections = self.pynn.Projection(pop[0], pop[1], connector_ie, synapse_ie, receptor_type = 'inhibitory', label = "inhibitory to excitatory") # from inhibitory to excitatory

		if writeflag:
			np.savetxt('resE_resI_weight.txt', E_I_connections.get('weight', format = 'array'), delimiter = ',')
			np.savetxt('resI_resE_weight.txt', I_E_connections.get('weight', format = 'array'), delimiter = ',')
		return [E_I_connections, I_E_connections]


class OutputNeuron(object):
	"""
	create neuron group for output decisions
	"""
	def __init__(self, pynn, config):
		self.config = config
		self.pynn = pynn
		self.dec_pops = []
		self.inh_pops = []
		self.num_pops = config['network'].as_int('num_dec')
		self.num_dec_neurons = config['network'].as_int('num_dec_neurons')
		self.num_inh_neurons = config['network'].as_int('num_inh_dec_neurons')
		for i in range(self.num_pops):
			self.dec_pops.append(self.pynn.Population(self.num_dec_neurons, self.pynn.IF_cond_exp()))
			self.inh_pops.append(self.pynn.Population(self.num_inh_neurons, self.pynn.IF_cond_exp()))
			self.dec_pops[i].record('spikes')
		# project each decision pops to coresponding inhi
		# self.prj_dec_inh = self.project_dec_inh()
		# # project each inh onto all other dec
		# self.prj_inh_dec = self.project_inh_dec()

	def get_spikecountmat(self):
		spikecountmat = []
		for i in range(self.num_pops):
			spikecountmat.append(self.dec_pops[i].get_data())
		return spikecountmat

	def project_dec_inh(self):
		w = self.config['network'].as_float('w_dec_inh')
		syn = self.pynn.StaticSynapse(weight = w, delay = 0.01)
		prj = []
		for i in range(self.num_pops):
			prj.append(self.pynn.Projection(self.dec_pops[i], self.inh_pops[i], self.pynn.AllToAllConnector(), syn, receptor_type = 'excitatory'))
		return prj
	
	def project_inh_dec(self):
		w = self.config['network'].as_float('w_inh_dec')
		syn = self.pynn.StaticSynapse(weight = w, delay = 0.01)
		prj = []
		for i in range(self.num_pops):
			for j in range(self.num_pops):
				if i == j:
					continue
				else:
					prj.append(self.pynn.Projection(self.inh_pops[i], self.dec_pops[j], self.pynn.AllToAllConnector(), syn, receptor_type = 'inhibitory'))
		return prj