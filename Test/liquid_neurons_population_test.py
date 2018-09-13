import numpy
import matplotlib
import pyNN.brian as sim
sim.setup()


Number_of_neurons_lsm = 125
Net_shape= (5,5,5) # 20x5=>layer, 6=>number of layers 
Number_of_neurons_I = 10
Net_shape_I= (10,1,1)

## === Define parameters ========================================================

'''
cell_params = {
    'tau_m'      : 20.0,   # (ms)
    'tau_syn_E'  : 2.0,    # (ms)
    'tau_syn_I'  : 4.0,    # (ms)
    'e_rev_E'    : 0.0,    # (mV)
    'e_rev_I'    : -70.0,  # (mV)
    'tau_refrac' : 2.0,    # (ms)
    'v_rest'     : -60.0,  # (mV)
    'v_reset'    : -70.0,  # (mV)
    'v_thresh'   : -50.0,  # (mV)
    'cm'         : 0.5}    # (nF)
dt         = 0.1           # (ms)
syn_delay  = 1.0           # (ms)
input_rate = 50.0          # (Hz)
simtime    = 1000.0        # (ms)
}
'''
## === Build Networks ========================================================
Population = {}
Population['Input'] = sim.Population(Number_of_neurons_I, sim.IF_curr_exp())
Population['Liquid'] = sim.Population(Number_of_neurons_lsm, sim.IF_curr_exp())