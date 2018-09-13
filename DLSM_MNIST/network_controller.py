"""
classed and functions to control network and classifier operation
"""
import sys
import time
import logging
import matplotlib.pyplot as plt
import numpy as np 
from pyNN.random import RandomDistribution, NumpyRNG

lg = logging.getLogger('network_controller')
lg.setLevel(logging.INFO)

import configobj
import numpy
from network import LSMBrain

class BrainController(object):
	"""
	set up the DLSM structure, control simulation, present stimuli, learn. 
	parameter:
	process -- training or testing
	"""
	def __init__(self, pynn, config, process, input_rate, item, w = None):
		# lg.info('setting up DLSM.')
		self.pynn = pynn
		self.config = config
		self.input_rate = input_rate
		self.brain = LSMBrain(pynn, config, process, input_rate[0, :]) # initialize the DLSM structure
		if self.config['network']['structure'] != 'no_inh':
			self.Inputprj_I, self.Inputprj_E = self.brain.wire_Input_to_Res()
		else:
			self.Inputprj_E = self.brain.wire_Input_to_Res()
		self.prj = self.brain.wire_Res_to_Dec(item, w) # a list of learning weight: loop order 1. dec_pop 2.exc & inh in reservior neuron
		self.process = process
		lg.info('Creating DLSM structure finished!')


	def prediction_pattern(self, duration, segment, class_ids = 'not used'):
		"""
		Present the input pattern and determine the network's choice.
		Returns the number of spikes produced in each decision population.
		
		Parameters:
		label -- label for the current present pattern
		class_ids -- list of strings containing all possible class labels (not used but necessary in classifiers)
		segment -- which segment need to be extract for decision
		"""
		self.pynn.run(duration)
		Dec_spikecounts = self.brain.Dec.get_spikecountmat()
		spike_count = []
		for i in range(len(Dec_spikecounts)):
			sum_spiketrain = 0
			for spiketrain in Dec_spikecounts[i].segments[segment].spiketrains:
				sum_spiketrain = sum_spiketrain + sum(spiketrain > 50)
			spike_count.append(sum_spiketrain)
		print spike_count
		predindex = np.argmax(spike_count)
		winner = class_ids[predindex]

		# fig_settings = {'lines.linewidth': 0.5, 'axes.linewidth': 0.5, 'axes.labelsize': 'small', 'legend.fontsize': 'small', 'font.size': 8} # define figure settings
		# plt.rcParams.update(fig_settings) # update figure settings
		# fig = plt.figure(1, figsize = (6, 8)) # create a figure
		# def plot_spiketrains(segment):
		# 	for spiketrain in segment.spiketrains:
		# 		y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
		# 		# print spiketrain
		# 		plt.plot(spiketrain, y, '.')
		# 		plt.ylabel(segment.name)
		# 		plt.setp(plt.gca().get_xticklabels(), visible = False)

		# plt.subplot(2, 1, 1)
		# plot_spiketrains(Dec_spikecounts[0].segments[0])
		# plt.subplot(2, 1, 2)
		# plot_spiketrains(Dec_spikecounts[1].segments[0])
		# # plt.xlabel("time (%s)" % Dec_spikecounts[0].segments[0].analogsignalarrays[0].times.units._dimensionality.string)
		# plt.setp(plt.gca().get_xticklabels(), visible = True)
		# plt.show()
		return winner, predindex

	def test_pattern(self, label, class_ids, item):
		try:
			batch_size = len(label)
		except TypeError:
			batch_size = 1

		duration = self.config['simulation'].as_float('epochtime')

		prediction = []
		w_output = self.get_weight_output_layer()
		for i in range(batch_size):
			if i != 0:
				self.adjust_input_rate(self.input_rate[i, :])
			pred, predindex = self.prediction_pattern(duration, i, class_ids)
			dec_correct = pred == label[i]
			print('Iteration No. %s, batch size %s, target is %s, prediction is %s, prediction is %s' %(i + 1, batch_size, int(label[i]), int(pred), ['WRONG', 'CORRECT'][dec_correct]))
			self.pynn.reset()
			
			## reset the connection weight for the DLSM
			# first reset the connection weight of final layer
			for j in range(len(self.prj)):
				self.prj[j].set(weight = w_output[j], delay = 0.1)
			# remember to reset the inhibitory weights for soft-winner-take-all structure
			# w = self.config['network'].as_float('w_dec_inh')
			# for k in range(len(self.brain.Dec.prj_dec_inh)):
			# 	self.brain.Dec.prj_dec_inh[k].set(weight = w, delay = 0.01)
			
			# w = self.config['network'].as_float('w_inh_dec')
			# for k in range(len(self.brain.Dec.prj_inh_dec)):
			# 	self.brain.Dec.prj_inh_dec[k].set(weight = w, delay = 0.01)


			# second reset the connection weight from input to reservoir neuron
			input_num = self.config['network'].as_int('num_input_neuron')
			if self.config['network']['structure'] != 'no_inh':
				w_I = np.loadtxt('input_resI_weight.txt', dtype = float, delimiter = ',')
				size = np.size(w_I, 1)
				for k in range(input_num):
					self.Inputprj_I[k].set(weight = w_I[k, :].reshape(1, size), delay = 0.5)
			w_E = np.loadtxt('input_resE_weight.txt', dtype = float, delimiter = ',')
			size = np.size(w_E, 1)
			for k in range(input_num):
				self.Inputprj_E[k].set(weight = w_E[k, :].reshape(1, size), delay = 0.5)

			# third reset the connection weight in reservoir neuron
			if not self.config['network']['structure'] == 'no_connection':
				if self.config['network']['structure'] != 'no_inh':
					w_ii = np.loadtxt('resI_resI_weight.txt', dtype = float, delimiter = ',')
					self.brain.Res.prj[1].set(weight = w_ii, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
					w_ei = np.loadtxt('resE_resI_weight.txt', dtype = float, delimiter = ',')
					self.brain.Res.prj[2].set(weight = w_ei, delay = 0.5, U = 0.04, tau_rec = 100.0, tau_facil = 1000)
					w_ie = np.loadtxt('resI_resE_weight.txt', dtype = float, delimiter = ',')
					self.brain.Res.prj[3].set(weight = w_ie, delay = 0.5, U = 0.04, tau_rec = 100.0, tau_facil = 1000)
				w_ee = np.loadtxt('resE_resE_weight.txt', dtype = float, delimiter = ',')
				self.brain.Res.prj[0].set(weight = w_ee, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)

			prediction.append(pred)

		return prediction




	def learn_pattern(self, label, class_ids, item):
		"""
		learn pattern
		parameters:
		label: a list of labels for training data: for label 0 should result output0 fire more frequently; for label 1 should result output1 fire more frequently
		class_ids: the MNIST data label
		item: current batch start training number
		"""
		try:
			batch_size = len(label)
		except TypeError:
			batch_size = 1

		duration = self.config['simulation'].as_float('epochtime')

		for i in range(batch_size):
			if i != 0:
				self.adjust_input_rate(self.input_rate[i, :])
			pred, predindex = self.prediction_pattern(duration, i, class_ids)
			dec_correct = pred == label[i]
			print('Iteration No. %s, batch size %s, target is %s, prediction is %s, prediction is %s' %(i + 1, batch_size, int(label[i]), int(pred), ['WRONG', 'CORRECT'][dec_correct]))
			w = self.get_weight_output_layer()
			self.pynn.reset()
			
			## reset the connection weight for the DLSM
			# first reset the connection weight of final layer
			if not dec_correct:
				labelindex = np.argwhere(class_ids == label[i])[0].item(0)
				self.adjust_weight_learningrule(labelindex, predindex, i, w)
				for j in range(len(self.prj)):
					if j != labelindex and j != predindex:
						self.prj[j].set(weight = w[j], delay = 0.1)
			else:
				for j in range(len(self.prj)):
					self.prj[j].set(weight = w[j], delay = 0.1)
			# remember to reset the inhibitory weights for soft-winner-take-all structure
			# w = self.config['network'].as_float('w_dec_inh')
			# for k in range(len(self.brain.Dec.prj_dec_inh)):
			# 	self.brain.Dec.prj_dec_inh[k].set(weight = w, delay = 0.01)
			
			# w = self.config['network'].as_float('w_inh_dec')
			# for k in range(len(self.brain.Dec.prj_inh_dec)):
			# 	self.brain.Dec.prj_inh_dec[k].set(weight = w, delay = 0.01)


			# second reset the connection weight from input to reservoir neuron
			input_num = self.config['network'].as_int('num_input_neuron')
			if self.config['network']['structure'] != 'no_inh':
				w_I = np.loadtxt('input_resI_weight.txt', dtype = float, delimiter = ',')
				size = np.size(w_I, 1)
				for k in range(input_num):
					self.Inputprj_I[k].set(weight = w_I[k, :].reshape(1, size), delay = 0.5)
			w_E = np.loadtxt('input_resE_weight.txt', dtype = float, delimiter = ',')
			size = np.size(w_E, 1)
			for k in range(input_num):
				self.Inputprj_E[k].set(weight = w_E[k, :].reshape(1, size), delay = 0.5)

			# third reset the connection weight in reservoir neuron
			if not self.config['network']['structure'] == 'no_connection':
				if self.config['network']['structure'] != 'no_inh':
					w_ii = np.loadtxt('resI_resI_weight.txt', dtype = float, delimiter = ',')
					w_ei = np.loadtxt('resE_resI_weight.txt', dtype = float, delimiter = ',')
					w_ie = np.loadtxt('resI_resE_weight.txt', dtype = float, delimiter = ',')
					if self.config['network']['structure'] != 'static':
						self.brain.Res.prj[1].set(weight = w_ii, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
						self.brain.Res.prj[2].set(weight = w_ei, delay = 0.5, U = 0.04, tau_rec = 100.0, tau_facil = 1000)
						self.brain.Res.prj[3].set(weight = w_ie, delay = 0.5, U = 0.04, tau_rec = 100.0, tau_facil = 1000)
					else:
						self.brain.Res.prj[1].set(weight = w_ii, delay = 0.2)
						self.brain.Res.prj[2].set(weight = w_ei, delay = 0.5)
						self.brain.Res.prj[3].set(weight = w_ie, delay = 0.5)
				w_ee = np.loadtxt('resE_resE_weight.txt', dtype = float, delimiter = ',')
				if self.config['network']['structure'] != 'static':
					self.brain.Res.prj[0].set(weight = w_ee, delay = 0.2, U = 0.5, tau_rec = 800.0, tau_facil = 0.01)
				else:
					self.brain.Res.prj[0].set(weight = w_ee, delay = 0.2)




			## if the weight recorded interval needed to be smaller than the training interval, uncomment the following code and modify the interval
			# record_interval = 4
			# if (item + i + 1)%record_interval == 0:
			# 	w = []
			# 	for j in range(len(self.prj)):
			# 		w.append(self.prj[j].get('weight', format = 'array'))
			# 	if self.process != 'no_inh':
			# 		for j in range(self.config['network'].as_int('num_dec')):
			# 			np.savetxt('resE_output%d_weight_No.%d.txt'%(j, item + i + 1), w[j * 2], delimiter = ',')
			# 			np.savetxt('resI_output%d_weight_No.%d.txt'%(j, item + i + 1), w[j * 2 + 1], delimiter = ',')
			# 	else:
			# 		for j in range(self.config['network'].as_int('num_dec')):
			# 			np.savetxt('resE_output%d_weight_No.%d.txt'%(j, item + i + 1), w[j], delimiter = ',')

			## the following programming plot the input neuron spike event. this is only for the purpose of debuging; comment this off normal experimental trial
			# Input_spikecounts = self.brain.Input.get_spikecountmat() # a list that contains the excitatory and inhibitory neuron spike event		
			# fig_settings = {'lines.linewidth': 0.5, 'axes.linewidth': 0.5, 'axes.labelsize': 'small', 'legend.fontsize': 'small', 'font.size': 8} # define figure settings
			# plt.rcParams.update(fig_settings) # update figure settings
			# fig = plt.figure(1, figsize = (6, 8)) # create a figure
			# def plot_spiketrains(segment):
			# 	for spiketrain in segment.spiketrains:
			# 		y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
			# 		# print spiketrain
			# 		plt.plot(spiketrain, y, '.')
			# 		plt.ylabel(segment.name)
			# 		plt.setp(plt.gca().get_xticklabels(), visible = False)

			# plt.subplot(2, 1, 1)
			# plot_spiketrains(Input_spikecounts[0].segments[i])
			# plt.subplot(2, 1, 2)
			# plot_spiketrains(Input_spikecounts[1].segments[i])
			# # plt.xlabel("time (%s)" % Dec_spikecounts[0].segments[0].analogsignalarrays[0].times.units._dimensionality.string)
			# plt.setp(plt.gca().get_xticklabels(), visible = True)
			# plt.show()

		w = self.get_weight_output_layer()
		return w

	def get_weight_output_layer(self):
		w = []
		for j in range(len(self.prj)):
			w.append(self.prj[j].get('weight', format = 'array'))
		return w

	def adjust_input_rate(self, input_rate):
		duration = self.config['simulation'].as_float('epochtime')
		for i in range(len(input_rate)):
			stimspike = RandomDistribution('uniform', low = 0, high = duration).next(int(duration * input_rate[i] * 1e-3))
			self.brain.Input.input_source[i].set(spike_times = stimspike)


	def adjust_weight_learningrule(self, label, pred, segment, w):
		lg.info('change weights accordingly')	
		rank_thresh = self.config['learningrule'].as_int('rank_thresh')
		rate_thresh = self.config['learningrule'].as_float('rate_thresh')
		duration = self.config['simulation'].as_float('epochtime')
		Res_spikecounts = self.brain.Res.get_spikecountmat() # a list that contains the excitatory and inhibitory neuron spike event
		
		# fig_settings = {'lines.linewidth': 0.5, 'axes.linewidth': 0.5, 'axes.labelsize': 'small', 'legend.fontsize': 'small', 'font.size': 8} # define figure settings
		# plt.rcParams.update(fig_settings) # update figure settings
		# fig = plt.figure(2, figsize = (6, 8)) # create a figure
		# def plot_spiketrains(segment):
		# 	for spiketrain in segment.spiketrains:
		# 		y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
		# 		# print spiketrain
		# 		plt.plot(spiketrain, y, '.')
		# 		plt.ylabel(segment.name)
		# 		plt.setp(plt.gca().get_xticklabels(), visible = False)

		# plt.subplot(2, 1, 1)
		# plot_spiketrains(Res_spikecounts[0].segments[segment])
		# plt.subplot(2, 1, 2)
		# plot_spiketrains(Res_spikecounts[1].segments[segment])
		# # plt.xlabel("time (%s)" % Dec_spikecounts[0].segments[0].analogsignalarrays[0].times.units._dimensionality.string)
		# plt.setp(plt.gca().get_xticklabels(), visible = True)
		# plt.show()

		pre_rates = []
		for j in range(len(Res_spikecounts)):
			for spiketrain in Res_spikecounts[j].segments[segment].spiketrains:
				pre_rates.append(sum(spiketrain > 50) / duration * 1e3)
		# find the n highest responding units, n < rank_thresh
		pre_rates = np.asarray(pre_rates)
		units_sortidx = np.argsort(pre_rates)
		units_sortidx = units_sortidx[::-1]
		# check whether rate_thresh or rank thresh is relevant
		# print pre_rates
		print pre_rates[units_sortidx[rank_thresh]]

		if pre_rates[units_sortidx[rank_thresh]] < rate_thresh:
			cutoff = np.searchsorted(pre_rates[units_sortidx[::-1]], rate_thresh)
			cutoff -= len(pre_rates)
			cutoff *= -1
		else:
			cutoff = rank_thresh

		unit_id_tup = units_sortidx[:cutoff]
		# set compute mode for dw
		w_min = self.config['learningrule'].as_float('w_min')
		w_max = self.config['learningrule'].as_float('w_max')
		delta_w_plus = self.config['learningrule'].as_float('delta_w_plus')
		delta_w_minus = self.config['learningrule'].as_float('delta_w_minus')
		num_exc = self.config['network'].as_int('num_exc')


		if self.config['network']['structure'] != 'no_inh':
			we_winner = w[pred * 2]
			wi_winner = w[pred * 2 + 1]
			we_target = w[label * 2]
			wi_target = w[label * 2 + 1]

			exc = unit_id_tup[unit_id_tup < num_exc]
			inh = unit_id_tup[unit_id_tup > num_exc -1]

			we_winner[exc, :] += -delta_w_minus
			we_winner[we_winner < w_min] = w_min
			wi_winner[inh - num_exc, :] += -delta_w_minus
			wi_winner[wi_winner < w_min] = w_min

			we_target[exc, :] += delta_w_plus
			we_target[we_target > w_max] = w_max
			wi_target[inh - num_exc, :] += delta_w_plus
			wi_target[wi_target > w_max] = w_max

			we_winner = self.prj[pred * 2].set(weight = we_winner, delay = 0.05)
			wi_winner = self.prj[pred * 2 + 1].set(weight = wi_winner, delay = 0.05)
			we_target = self.prj[label * 2].set(weight = we_target, delay = 0.05)
			wi_target = self.prj[label * 2 + 1].set(weight = wi_target, delay = 0.05)

		else:
			exc = unit_id_tup[unit_id_tup < num_exc]
			we_winner = w[pred]
			we_target = w[label]

			we_winner[exc, :] += -delta_w_minus
			we_winner[we_winner < w_min] = w_min

			we_target[exc, :] += delta_w_plus
			we_target[we_target > w_max] = w_max

			self.prj[pred].set(weight = we_winner, delay = 0.05)
			self.prj[label].set(weight = we_target, delay = 0.05)
		return



# =============== plot figure ===============
			

# fig_settings = {'lines.linewidth': 0.5, 'axes.linewidth': 0.5, 'axes.labelsize': 'small', 'legend.fontsize': 'small', 'font.size': 8} # define figure settings
# plt.rcParams.update(fig_settings) # update figure settings
# fig = plt.figure(i, figsize = (6, 8)) # create a figure
# def plot_spiketrains(segment):
# 	for spiketrain in segment.spiketrains:
# 		y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
# 		# print spiketrain
# 		plt.plot(spiketrain, y, '.')
# 		plt.ylabel(segment.name)
# 		plt.setp(plt.gca().get_xticklabels(), visible = False)

# plt.subplot(2, 1, 1)
# plot_spiketrains(Res_spikecounts[0].segments[0])
# plt.subplot(2, 1, 2)
# plot_spiketrains(Res_spikecounts[1].segments[0])
# # plt.xlabel("time (%s)" % Dec_spikecounts[0].segments[0].analogsignalarrays[0].times.units._dimensionality.string)
# plt.setp(plt.gca().get_xticklabels(), visible = True)
# plt.show()
