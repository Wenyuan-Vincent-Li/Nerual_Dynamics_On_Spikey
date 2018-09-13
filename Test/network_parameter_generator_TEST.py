import numpy as np
import random

N_e = 10 # number of excitatory neurons
N_i = 10 # number of inhibitatory neurons
Exc_in = 10 # number of excitatory inputs
Inh_in = 10 # number of inhibitatory inputs

P_c = 0.1 # scaling factor of connection probability; make sure the largest p (i.e. p_ii if (N_e > N_i)) is smaller than 1
p_ee = P_c
p_ie = 2 * P_c
p_ei = N_e / N_i * P_c
p_ii = 2 * N_e / N_i * P_c


def generate_network_connection_array(row,col,probability):
	result = np.random.binomial(1, probability, row*col)
	result = np.asarray(result).reshape((row,col))
	return result

E_E_connections = generate_network_connection_array(N_e,N_e,p_ee)
I_I_connections = generate_network_connection_array(N_i,N_i,p_ii)
E_I_connections = generate_network_connection_array(N_e,N_i,p_ei)
I_E_connections = generate_network_connection_array(N_i,N_e,p_ie)

Input_connections = generate_network_connection_array(Exc_in + Inh_in, N_e + N_i, 0.5)

print Input_connections


# np.savetxt('connection_matrix/Input_connections.txt', Input_connections, fmt='%d', delimiter=',')

# # Ie_E_connections = generate_network_connection_array(Exc_in,N_e,0.5)
# # Ii_E_connections = generate_network_connection_array(Inh_in,N_e,0.5)
# # Ie_I_connections = generate_network_connection_array(Exc_in,N_i,0.5)
# # Ii_I_connections = generate_network_connection_array(Inh_in,N_i,0.5)


# np.savetxt('connection_matrix/E_E_connections.txt', E_E_connections, fmt='%d', delimiter=',')
# np.savetxt('connection_matrix/I_I_connections.txt', I_I_connections, fmt='%d', delimiter=',')
# np.savetxt('connection_matrix/E_I_connections.txt', E_I_connections, fmt='%d', delimiter=',')
# np.savetxt('connection_matrix/I_E_connections.txt', I_E_connections, fmt='%d', delimiter=',')

# np.savetxt('Ie_E_connections.txt', Ie_E_connections, fmt='%d', delimiter=',')
# np.savetxt('Ii_E_connections.txt', Ii_E_connections, fmt='%d', delimiter=',')
# np.savetxt('Ie_I_connections.txt', Ie_I_connections, fmt='%d', delimiter=',')
# np.savetxt('Ii_I_connections.txt', Ii_I_connections, fmt='%d', delimiter=',')








# Create an array connector based on the probability above 



#connector = ArrayConnector(connections)

# connector_ee = sim.FixedProbabilityConnector(p_connect = p_ee) # connector algorithm 
# connector_ie = sim.FixedProbabilityConnector(p_connect = p_ie)
# connector_ei = sim.FixedProbabilityConnector(p_connect = p_ei)
# connector_ii = sim.FixedProbabilityConnector(p_connect = p_ii)

