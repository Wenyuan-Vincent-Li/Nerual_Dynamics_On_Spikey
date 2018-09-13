# Neural Dynamics On Spikey

This project aims study the neural dynamics phenomenon on a neuromorphic chip Spikey. We've
published a paper on [Neural Computation](https://www.mitpressjournals.org/loi/neco), 
you can download our paper [here](https://www.mitpressjournals.org/doi/abs/10.1162/neco_a_01103).


## Introduction

The extreme complexity of the brain has attracted the attention of neuroscientists 
and other researchers for a long time. More recently, the neuromorphic hardware 
has matured to provide a new powerful tool to study neuronal dynamics. In this project, 
we study neuronal dynamics using different settings on a neuromorphic chip built 
with flexible parameters of neuron models. Our unique setting in the network of 
leaky integrate-andfire (LIF) neurons is to introduce a weak noise environment. 
We observed three different types of collective neuronal activities, or phases, 
separated by sharp boundaries, or phase transitions. From this, 
we construct a rudimentary phase diagram of neuronal dynamics and demonstrate that 
a noise-induced chaotic phase (N-phase), which is dominated by neuronal avalanche 
activity (intermittent aperiodic neuron firing), emerges in the presence of noise 
and its width grows with the noise intensity. The dynamics can be manipulated in 
this N-phase. Our results and comparison with clinical data is consistent with the 
literature and our previous work showing that healthy brain must reside in the N-phase.

## Settings

We constructed a recurrent network of neurons with sparse and random connections on 
the Spikey chip. Two types of recordings were extracted from the emulation, the membrane
potential V<sub>m</sub> of a randomly chosen neuron and the “spike train” data as shown in the
following Figure.

<img src="/Plots/Settings.png">

## Results

### Emulation Results from the Neuromorphic Chip

Following figure shows three typical neuronal dynamic behaviors in the neural 
network. The raster plots of spike train data and membrane potential recordings 
are shown in the top and bottom rows, respectively. (a) Constant oscillating firing 
behavior presents in the seizure state/C-phase with a high activity correlation. 
(b) Normal state/N-phase behavior shows intermittent firing events with relatively 
low activity correlation. (c) None or only a few firing activities are present 
in the coma state/T-phase with extremely low activity correlation.

### Comparison with Brain Slice Recordings.

Following figure shows the comparison between human brain slice data and emulation 
data in the coma-like (top row), the normal (middle row), and the seizure-like 
(bottom row) states. In the coma-like state, we did not observe any firing 
activities at both membrane potential recordings (see top row in panel a). 
Their corresponding power spectrum shows a sharp drop at low frequencies and the 
typical normalized power spectrum in the frequency domain is below 10<sup>-3</sup> 
in the frequency domain of measurement (see top row, panel b). In the normal state, 
we observe the intermittent firing activities in both systems (see the middle row, 
panel a). Their power spectrum shows a 1/f<sup>α</sup> (a linear line on 
log-log plot) behavior (see the middle row in panel b). In the seizure-like state, 
we observe the oscillatory behavior from both membrane potential recordings 
(see the bottom row in panel a) and find that the power spectra are both 1/f<sup>α</sup>
superimposed by peaks (see the bottom row in panel b).

### Phase Diagram

* Following shows the phase diagram of neuronal dynamics constructed from power spectrum analysis. 
(a) The phase diagram shows the dependence of noise versus threshold potential. 
In the deterministic limit (no noise present), the N-phase collapses onto a sharp 
boundary between the T-phase and C-phase (vertical dashed line at around −60.5 mV), 
as predicted by the previous theory. The noiseinduced phase (N-phase) emerges 
when noise is present in the system. The noise is an essential parameter in this 
N-phase picture. The higher the noise intensity is, the wider the N-phase (in the sense 
of threshold range) becomes. The insets in each phase show the typical membrane 
potential behavior on different timescales. Zoomed panel (b) shows the sharp transition 
in the deterministic limit between the T-phase and the C-phase. A clear difference 
in both membrane potential recordings and their corresponding power spectra is seen, 
even though there is only a very small change from V<sub>th</sub> = −60.6 mV to 
V<sub>th</sub> = −60.5 mV in the case of when no noise is present.

* Following shows An alternative way to construct the phase diagram. Based on the 
correlation parameter C<sub>sync</sub> (detailed in the paper), a brain phase 
diagram was constructed similar to that in the above Figure.
