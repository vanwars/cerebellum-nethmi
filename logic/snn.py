from brian2 import PoissonGroup, Network, NeuronGroup, SpikeMonitor, \
    Synapses, mV, ms, Hz, second
import brian2.numpy_ as np
import numpy
from formulas import FORMULAS
from constants import *


class SpikingNeuralNetwork():
    def __init__(self):

        # 1. create the input neurons:
        self.poisson_group = PoissonGroup(
            input_neuron_size, rates=np.zeros(input_neuron_size)*Hz, name='PG'
        )

        # 2. create the excitatory neuron group
        self.exitatory_neuron_group = NeuronGroup(
            n_e, FORMULAS.neuron_e, threshold='v>v_thresh_e', refractory=5*ms,
            reset='v=v_reset_e', method='euler', name='EG'
        )
        self.exitatory_neuron_group.v = v_rest_e - 20.*mV

        # 3. create the inhibitory neuron group
        self.inhibitory_neuron_group = NeuronGroup(
            n_i, FORMULAS.neuron_i, threshold='v>v_thresh_i', refractory=2*ms, 
            reset='v=v_reset_i', method='euler', name='IG'
        )
        self.inhibitory_neuron_group.v = v_rest_i - 20.*mV

        # 4. connect the poisson neurons one-to-all to the excitatory neurons with plastic connections: 
        self.poisson_to_excitory_synapses = Synapses(
            self.poisson_group, self.exitatory_neuron_group, FORMULAS.stdp,
            on_pre=FORMULAS.pre, on_post=FORMULAS.post, method='euler', name='S1'
        )
        self.poisson_to_excitory_synapses.connect()
        self.poisson_to_excitory_synapses.w = 'rand()*gmax'  # random weights initialisation
        self.poisson_to_excitory_synapses.lr = 1  # enable stdp

        # 5. Connect the excitatory neurons one-to-one to inhibitory neurons:
        self.excitory_to_inhibitory_synapses = Synapses(
            self.exitatory_neuron_group, self.inhibitory_neuron_group, 'w : 1',
            on_pre='ge += w', name='S2'
        )
        self.excitory_to_inhibitory_synapses.connect(j='i')
        self.excitory_to_inhibitory_synapses.delay = 'rand()*10*ms'
        # very strong fixed weights to ensure corresponding inhibitory neuron will always fire
        self.excitory_to_inhibitory_synapses.w = 3
        
        # 6. inhibitory neurons one-to-all-except-one excitatory neurons
        self.inhibitory_to_excitory_synapses = Synapses(
            self.inhibitory_neuron_group, self.exitatory_neuron_group, 'w : 1',
            on_pre='gi += w', name='S3')
        self.inhibitory_to_excitory_synapses.connect(condition='i!=j')
        self.inhibitory_to_excitory_synapses.delay = 'rand()*5*ms'
        # weights are selected in such a way as to maintain a balance between excitation and ibhibition
        self.inhibitory_to_excitory_synapses.w = .03

        self.net = Network(
            self.poisson_group,
            self.exitatory_neuron_group,
            self.inhibitory_neuron_group,
            self.poisson_to_excitory_synapses,
            self.excitory_to_inhibitory_synapses,
            self.inhibitory_to_excitory_synapses
        )
        self.net.run(0*second)

    def __getitem__(self, key):
        return self.net[key]

    def train(self, X, epoch=1):
        self.net['S1'].lr = 1  # stdp on

        for ep in range(epoch):
            for idx in range(len(X)):

                if (idx % 100 == 0):
                    print(str(epoch), "_", str(idx))
                
                # active mode
                self.net['PG'].rates = X[idx].ravel()*Hz
                self.net.run(0.35*second)

                # passive mode
                self.net['PG'].rates = np.zeros(input_neuron_size)*Hz
                self.net.run(0.15*second)

    
    
    # not sure what this function does exactly. I think it makes spike
    # trains out of the test data and saves the data as a python dictionary.
    # However, I'm unclear re: whether this is an evaluation.
    def evaluate(self, X, test_labels):
        self.net['S1'].lr = 0  # stdp off
        store_idx = numpy.zeros(10).astype(int)
        for idx in range(len(X)):

            print(str(idx), str(store_idx[idx]), str(store_idx[test_labels[idx]]))

            # rate monitor to count spikes
            mon = SpikeMonitor(self.net['EG'])
            self.net.add(mon)

            # active mode (larger active mode)
            self.net['PG'].rates = X[idx].ravel()*Hz
            self.net.run(5*second)

            file_name = "MNIST_epoch1_eineuron1000_train1to9/spike_dict_label_" + \
                str(store_idx[idx]) + "_instance_" + \
                str(store_idx[test_labels[idx]])+".txt"
            print(file_name)
            f = open(file_name, "w")
            f.write(str(mon.spike_trains()))
            f.close()
            store_idx[test_labels[idx]] += 1

