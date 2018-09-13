import pyNN.brian as sim
import numpy as np 


column = 4 # column of plastic synapse
row = 4 # row of plastic synapse
weightPlastic = 0.0 # weight of plastic synapse
noSpikePairs = 20 # number of spike pairs
timingPrePostPlastic = 1.0 # timing between pre- and postsynaptic spikes at plastic synapse in ms
intervalPairs = 100.0 # time interval between presynaptic spikes in ms

noStim = 3 # number of synapses to stimulate spiking of postsynaptic neuron
weightStim = 8.0 # weight of stimulating synapses
timingPrePostStim = 0.3 # limit of precision of spiking in ms
stimulusOffset = 100.0 # offset from beginning and end of emulation in ms (should be larger than timingPrePostPlastic)

# prepare stimuli
stimulus = np.arange(stimulusOffset, (noSpikePairs - 0.5) * intervalPairs + stimulusOffset, intervalPairs)
stimulusPlastic = stimulus + timingPrePostStim - timingPrePostPlastic
# print stimulusPlastic
# print stimulus, (noSpikePairs - 0.5) * intervalPairs
# assert(len(stimulus) == noSpikePairs)
sim.setup()

# create postsynaptic neuron
neuron = sim.Population(1, sim.IF_curr_exp())


spikeSourceStim = None
spikeSourcePlastic = None
# place stimulating synapses above plastic synapse
if row < noStim:
	if row > 0:
		dummy = sim.Population(row, sim.SpikeSourceArray)
	spikeSourcePlastic = sim.Population(1, sim.SpikeSourceArray, {'spike_times' : stimulusPlastic})

# create stimulating inputs
spikeSourceStim = sim.Population(noStim, sim.SpikeSourceArray, {'spike_times' : stimulus})

# place stimulating synapses below plastic synapse
if row >= noStim:
	if row >noStim:
		dummy = sim.Population(row - noStim, sim.SpikeSourceArray, {'spike_times' : []})
	spikeSourcePlastic = sim.Population(1, sim.SpikeSourceArray, {'spike_times' : stimulusPlastic})
assert(spikeSourceStim != None)
assert(spikeSourcePlastic != None)

# configure stdp
stdp = sim.STDPMechanism(weight = 0.2,  # this is the initial value of the weight
	timing_dependence = sim.SpikePairRule(tau_plus = 20.0, tau_minus = 20.0,
		A_plus = 0.01, A_minus = 0.012),\
	weight_dependence = sim.AdditiveWeightDependence(w_min = 0, w_max = 0.04))

# connect stimulus
sim.Projection(spikeSourceStim, neuron, sim.AllToAllConnector(), sim.StaticSynapse(weight = 0.04, delay = timingPrePostStim), receptor_type = 'excitatory')

# create plastic synapse
prj = sim.Projection(spikeSourcePlastic, neuron, sim.AllToAllConnector(), stdp)
weightBefore = prj.get('weight', format = 'list')
prj.set(weight = 0.15)
print weightBefore
neuron.record('spikes')

lastInputSpike = np.max(np.concatenate((stimulus, stimulusPlastic)))
runtime = lastInputSpike + stimulusOffset

sim.run(runtime)

weightAfter = prj.get('weight', format = 'list')
print weightAfter
result = neuron.get_data()

sim.end()

spikeTimes = result.segments[0].spiketrains[0]

# analysis
print 'Number of stimulating / presynaptic / postsynaptic spikes:', len(stimulus), len(stimulusPlastic), len(spikeTimes)

if len(stimulusPlastic) != len(spikeTimes):
    print 'Not each presynaptic spike has a single postsynaptic partner!'
    print '\nstimulating spikes:'
    print stimulus
    print '\npresynaptic spikes:'
    print stimulusPlastic
    print '\npostsynaptic spikes:'
    print spikeTimes
    exit()
timingMeasured = np.mean(spikeTimes - stimulusPlastic)
print 'Time interval between pre- and postsynaptic spike (is / should / limit):', timingMeasured, '/', timingPrePostPlastic, '/', spikePrecision
if abs(timingMeasured - timingPrePostPlastic) > spikePrecision:
    print 'Time interval between pre- and postsynaptic deviates from expectation. Adjust delay parameter.'
print 'Synaptic weight before / after emulation (in digital hardware values):', weightPlastic, weightAfter
