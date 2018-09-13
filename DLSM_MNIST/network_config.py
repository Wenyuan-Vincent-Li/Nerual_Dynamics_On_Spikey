"""
create the network configuration for DLSM
"""

import configobj
import StringIO

configstr = StringIO.StringIO("""
	[network]
	# specify the structure of the implemented DLSM structure, include 'dynamic', 'static', 'no_connection' and 'no_inh'
	structure = 'dynamic'
	
	# input neurons
	num_input_neuron = 64 # number of neurons for input source
	
	# reservior neurons
	num_exc = 75 # number of excitatory neurons
	label_exc = 'excitatory neuron'
	num_inh = 25 # number of inhibitatory neurons
	label_inh = 'inhibitory neuron'
	p_ee = 0.2 # default value 0.1
	p_ei = 0.2 # default value 0.2
	p_ie = 0.3 # default value 0.3
	p_ii = 0.4 # default value 0.6
	v_th = -15.0
	
	# output neurons
	num_dec_neurons = 10 # number for output neurons for each group
	num_dec = 2 # number of decisions need to be made
	num_inh_dec_neurons = 8 # number of inhibitory neurons per decision group
	w_dec_inh = 0.0001 # weight from dec to inh in output neurons 0.01
	w_inh_dec = 0.0001 # weight form inh to dec in output neurons 0.1

	[simulation]
	epochtime = 200

	[learningrule]
	w_max = 0.3 # maximum weight
	w_min = 0.0001 # minimum weight
	delta_w_plus = 0.001
	delta_w_minus = 0.001
	rate_thresh = 15 # threshold in spike rate above which to modify synapses
	rank_thresh = 10 # threshold in firing rate rank until which to modify

	[data]
	training_datasize = 500
	testing_datasize = 100

	""")
software_example_config = configobj.ConfigObj(configstr)