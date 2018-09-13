import numpy as np
import random


# adjacent connection matrix generator
Pre_no = 10
Post_no = 25
Prob = 0.5
filename = 'input1_i_res.txt'


def generate_network_connection_array(row, col, probability):
	result = np.random.binomial(1, probability, row * col)
	result = np.asarray(result).reshape((row, col))
	return result

connections = generate_network_connection_array(Pre_no, Post_no, Prob)
# print np.size(connections)
np.savetxt(filename, connections, fmt='%d', delimiter=',')