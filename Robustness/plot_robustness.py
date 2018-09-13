import numpy as np 
import matplotlib.pyplot as plt

dynamics = np.loadtxt("robustness_exp=1", delimiter = ',')
no_connection = np.loadtxt("robustness_exp=2", delimiter = ',')
static = np.loadtxt("robustness_exp=3", delimiter = ',')
no_inh = np.loadtxt("robustness_exp=4", delimiter = ',')
# print data[:, 1]

# =========plot figure=============
# fig_settings = {'lines.linewidth': 0.5, 'axes.linewidth': 0.5, 'axes.labelsize': 'small', 'legend.fontsize': 'small', 'font.size': 8} # define figure settings
# plt.rcParams.update(fig_settings) # update figure settings
fig = plt.figure(1, figsize = (5, 5)) # create a figure

# plt.plot(dynamics[:, 0], dynamics[:, 1], 'r*', label = 'dynamics', no_connection[:, 0], no_connection[:, 1], 'bs', label = 'no_connections', static[:, 0], static[:, 1], 'g^', label = 'static')
plt.plot(dynamics[:, 0], np.log10(dynamics[:, 1]), 'r*-', label = 'dynamic synapses')
plt.plot(no_connection[:, 0], np.log10(no_connection[:, 1]), 'bs-', label = 'no connection')
plt.plot(static[:, 0], np.log10(static[:, 1]), 'g^-', label = 'static synapses')
plt.plot(no_inh[:, 0], np.log10(no_inh[:, 1]), 'y.-', label = 'no inhibitory')
plt.ylabel("Average firing rate (log(Hz))")
plt.xlabel("Input signal weight")
plt.legend()
plt.show()
# fig.savefig('no_explosion.png')