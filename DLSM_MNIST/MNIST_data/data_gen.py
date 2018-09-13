"""
generating training and testing data sets for MNIST  data

"""
import logging
import numpy as np 
import os
lg = logging.getLogger('data_gen')
lg.setLevel(logging.INFO)


class Data_Generation(object):
	def __init__(self):
		return

	def generating_data(self, dataset):
		data = None
		initial = 0
		path = os.path.dirname(__file__)
		filename_postfix = ""
		for i in dataset:
			filename = os.path.join(path, 'lambda_parameter_%d.txt'%i)
			current_data = np.loadtxt(filename, delimiter = ',')
			filename_postfix += str("_%d"%i) 
			if initial == 0:
				data = current_data
				initial += 1
			else:
				data = np.vstack((data, current_data))

		#shuffle the data and generating the training and testing data
		data = data[np.random.permutation(data.shape[0]), :]
		training_data = data[:int(np.size(data, 0) * 0.6), :]
		testing_data = data[int(np.size(data, 0) * 0.4) : , :]
		np.savetxt('training_data' + filename_postfix + '.txt', training_data, delimiter = ',')
		np.savetxt('testing_data' + filename_postfix + '.txt', testing_data, delimiter = ',')
		lg.info('Generating training and testing data')