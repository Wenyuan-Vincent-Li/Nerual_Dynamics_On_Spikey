"""
generating structure i.e. connection matrix for DLSM

"""
import logging
import random as rm
import numpy as np 
lg = logging.getLogger('structure_gen')
lg.setLevel(logging.INFO)

class Structure_Generation(object):
	def __init__(self, config):
		self.config = config

	def generating_structure(self):
		# generating input to resvoir neuron connection matrix
		lg.info('Generating connection matrix from input to resvior neuron')
		num_exc = self.config['network'].as_int('num_exc')
		num_inh = self.config['network'].as_int('num_inh')
		num_input_neuron = self.config['network'].as_int('num_input_neuron')
		self.connection_matrix(num_input_neuron, num_exc, 0.5, 'input_resE.txt')
		if num_inh != 0:
			self.connection_matrix(num_input_neuron, num_inh, 0.5, 'input_resI.txt')

		# generating resvoir neuron connection matrix
		if self.config['network']['structure'] != 'no_connection':
			lg.info('Generating connection matrix in resvior neuron')
			p_ee = self.config['network'].as_float('p_ee')
			self.connection_matrix(num_exc, num_exc, p_ee, 'resE_resE.txt')
			if num_inh != 0:
				p_ei = self.config['network'].as_float('p_ei')
				p_ie = self.config['network'].as_float('p_ie')
				p_ii = self.config['network'].as_float('p_ii')

				self.connection_matrix(num_exc, num_inh, p_ei, 'resE_resI.txt')
				self.connection_matrix(num_inh, num_exc, p_ie, 'resI_resE.txt')
				self.connection_matrix(num_inh, num_inh, p_ii, 'resI_resI.txt')

		# generatign rervoir neuron to output neuron connection matrix
		lg.info('Generating connection matrix from resvior neuron to output')
		num_output = self.config['network'].as_int('num_dec')
		num_dec_neurons = self.config['network'].as_int('num_dec_neurons')
		for i in range(num_output):
			self.connection_matrix(num_exc, num_dec_neurons, 0.8, 'resE_output%d.txt'%i)
			if num_inh != 0:
				self.connection_matrix(num_inh, num_dec_neurons, 0.8, 'resI_output%d.txt'%i)

	def connection_matrix(self, pre, post, p, filename):
		'''
		This function geneerating a connection matrix with probability p. 1 represents there is a connection and 0 represents there is not a connection
		parameter:
		pre -- number of presynaptic neuron 
		post -- number of postsynaptic neuron
		probability -- connection probability
		filename -- saving filename
		'''
		matrix = np.random.binomial(1, p, pre * post)
		matrix = np.asarray(matrix).reshape((pre, post))
		np.savetxt(filename, matrix, fmt='%d', delimiter=',')