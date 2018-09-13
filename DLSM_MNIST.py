"""
This DLSM_MNIST file construct a dynamic liquid state machine structure to train and test on MNIST data set.

The input contains 64 neurons represent 64 coefficient extract using neural autoencoder. coefficient>0: excitatory; coefficient<0: inhibitory

The output decision neurons are soft-winner-take-all structure. 
"""
import sys
import os.path
import time
import logging
import numpy as np

sys.path.append('/Users/wenyuan/Documents/PyNN')
logging.basicConfig()
lg = logging.getLogger('DLSMtest_loging')
lg.setLevel(logging.INFO)
try:
	import pyNN.brian as sim
except ImportError, e:
	print('ImportError:{}.\n'.format(e.message))
	sys.exit(1)

from DLSM_MNIST.network_config import software_example_config as config # this py file contains all the configuration for the DLSM structure
import DLSM_MNIST.network_controller as netcontrol


# ========== changes for network config ========
structure = 'dynamic' # specify the structure of the implemented DLSM structure, include 'dynamic', 'static', 'no_connection' and 'no_inh'
config['network']['structure'] = structure

if structure == 'no_inh':
	config['network']['num_exc'] = config['network'].as_int('num_exc') + config['network'].as_int('num_inh')
	config['network']['num_inh'] = 0


# ========== control precedure ==========
Generating_data = False # generating training and testing data
MNIST_dataset = [0, 1, 2] # define which MNIST data you want to use for training and testing
config['network']['num_dec'] = len(MNIST_dataset)

filename_postfix = ''
for i in MNIST_dataset:
	filename_postfix += str("_%d"%i) 

Generating_structure = False # generating or regenerating necessary structure information/ connection matrix

training = True
start_training = 300 # define start and end training example
end_training = 400
step_training = 20 # define after how many training set run we will jump out of simulation run and clear the data segments

testing = True
start_testing = 0
end_testing = 40
step_testing = 40 # define after how many training set run we will jump out of simulation run and clear the data segments
training_weight_no = np.arange(start_training + step_training, end_training + 1, step_training) # define which training weight you want to use for testing
# training_weight_no = [200]


evaluating = True
start_evaluating = 0
end_evaluating = end_testing
evaluating_no = None
# ========== generating necessary data ==========
"""
This section check if the needed data exsits in the current folder. If not it will help generating the necessary data set including training data set, testing data set, connection matrix.

"""
if Generating_data:
	import DLSM_MNIST.MNIST_data.data_gen as data_gen # this py file generating training and testing examples
	DG = data_gen.Data_Generation()
	DG.generating_data(MNIST_dataset)

if Generating_structure:
	import DLSM_MNIST.structure_gen as structure_gen # this py file generating connection matrxi for DLSM structure
	SG = structure_gen.Structure_Generation(config)
	SG.generating_structure()


# ========== training DLSM ==========
"""
This section perform the training process for DLSM

"""


if training:
	training_data = np.loadtxt('training_data' + filename_postfix + '.txt', dtype = float, delimiter = ',')
	data_seq = np.arange(start_training, end_training, step_training)
	w = None
	## ========== setup the classifier network ==========
	for index, item in enumerate(data_seq):
		if item + step_training > end_training:
			batch_end = end_training
		else:
			batch_end = item + step_training

		input_rate = training_data[item : batch_end, : 64]
		label = training_data[item : batch_end, 64]
		print ('Training batch %d, total batch %d.'%(index + 1, len(data_seq)))
		sim.setup()
		bc = netcontrol.BrainController(sim, config, 'training', input_rate, item, w)
		w = bc.learn_pattern(np.int_(label), MNIST_dataset, item)
		sim.end()
		# # save the trained weight
		if config['network']['structure'] != 'no_inh':
			for i in range(config['network'].as_int('num_dec')):
				np.savetxt('resE_output%d_weight_No.%d.txt'%(i, batch_end), w[i * 2], delimiter = ',')
				np.savetxt('resI_output%d_weight_No.%d.txt'%(i, batch_end), w[i * 2 + 1], delimiter = ',')
		else:
			for i in range(config['network'].as_int('num_dec')):
				np.savetxt('resE_output%d_weight_No.%d.txt'%(i, batch_end), w[i], delimiter = ',')

# ========== testing DLSM ==========
"""
This section perform the testing process for DLSM

"""

if testing:
	testing_data = np.loadtxt('testing_data' + filename_postfix + '.txt', dtype = float, delimiter = ',')
	data_seq = np.arange(start_testing, end_testing, step_testing)
	for i in range(len(training_weight_no)):
		print('Testing process for training weight No.%d'%training_weight_no[i])
		prediction = []
		for index, item in enumerate(data_seq):
			if item + step_testing > end_testing:
				batch_end = end_testing
			else:
				batch_end = item + step_testing
			input_rate = testing_data[item : batch_end, : 64]
			label = testing_data[item : batch_end, 64]
			print ('Testing batch %d, total batch %d.'%(index + 1, len(data_seq)))
			sim.setup()
			bc = netcontrol.BrainController(sim, config, 'testing', input_rate, item, training_weight_no[i])
			pred = bc.test_pattern(np.int_(label), MNIST_dataset, item)
			sim.end()
			prediction = prediction + pred
		## save prediction
		np.savetxt('testing_data' + filename_postfix + '_prediction_No%d.txt'%training_weight_no[i], prediction, delimiter = ',')


# ========== evaluating DLSM ==========
"""
This section evaluating DLSM performance

"""
if evaluating:
	testing_data = np.loadtxt('testing_data' + filename_postfix + '.txt', dtype = float, delimiter = ',')
	testing_label = testing_data[start_evaluating : end_evaluating, 64]
	if evaluating_no == None:
		evaluating_no = training_weight_no
	accuracy = []
	for i in range(len(evaluating_no)):
		prediction = np.loadtxt('testing_data' + filename_postfix + '_prediction_No%d.txt'%evaluating_no[i], dtype = float, delimiter = ',')
		total = len(prediction)
		acc = float(sum(prediction[start_evaluating : end_evaluating] == testing_label)) / total
		print ('the prediction accuracy performance is %.2f%% for training weight No.%d' % (acc * 100, evaluating_no[i]))
		accuracy.append(acc)