from brian2 import mV, ms


input_neuron_size = 28*28
n_e = 1000  # e - excitatory
n_i = n_e  # i - inhibitory

v_rest_e = -60*mV  # v - membrane potential
v_reset_e = -65.*mV
v_thresh_e = -52.*mV

v_rest_i = -60*mV
v_reset_i = -45.*mV
v_thresh_i = -40.*mV

taupre = 20*ms
taupost = taupre
gmax = .05  # .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax