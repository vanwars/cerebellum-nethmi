from brian2 import PoissonGroup, Network, NeuronGroup, SpikeMonitor, \
    StateMonitor, PopulationRateMonitor, Synapses, mV, ms, Hz, second
import brian2.numpy_ as np
import numpy

# help(Network)

# Constants:
n_input = 28*28  # input layer
n_e = 1000  # e - excitatory
n_i = n_e  # i - inhibitory

v_rest_e = -60.*mV  # v - membrane potential
v_reset_e = -65.*mV
v_thresh_e = -52.*mV

v_rest_i = -60.*mV
v_reset_i = -45.*mV
v_thresh_i = -40.*mV

taupre = 20*ms
taupost = taupre
gmax = .05  # .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

# Apre and Apost - presynaptic and postsynaptic traces, lr - learning rate
stdp = '''w : 1
        lr : 1 (shared)
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)'''
pre = '''ge += w
        Apre += dApre
        w = clip(w + lr*Apost, 0, gmax)'''
post = '''Apost += dApost
        w = clip(w + lr*Apre, 0, gmax)'''

print(stdp)
print(pre)
print(post)


class CerebellarCircuitModel():

    def __init__(self, debug=False):
        app = {}

        # input images as rate encoded Poisson generators
        app['PG'] = PoissonGroup(
            n_input, rates=np.zeros(n_input)*Hz, name='PG')

        # excitatory group
        neuron_e = '''
            dv/dt = (ge*(0*mV-v) + gi*(-100*mV-v) + (v_rest_e-v)) / (100*ms) : volt
            dge/dt = -ge / (5*ms) : 1
            dgi/dt = -gi / (10*ms) : 1
            '''
        app['EG'] = NeuronGroup(n_e, neuron_e, threshold='v>v_thresh_e',
                                refractory=5*ms, reset='v=v_reset_e', method='euler', name='EG')
        app['EG'].v = v_rest_e - 20.*mV

        if (debug):
            app['ESP'] = SpikeMonitor(app['EG'], name='ESP')
            app['ESM'] = StateMonitor(
                app['EG'], ['v'], record=True, name='ESM')
            app['ERM'] = PopulationRateMonitor(app['EG'], name='ERM')

        # ibhibitory group
        neuron_i = '''
            dv/dt = (ge*(0*mV-v) + (v_rest_i-v)) / (10*ms) : volt
            dge/dt = -ge / (5*ms) : 1
            '''
        app['IG'] = NeuronGroup(n_i, neuron_i, threshold='v>v_thresh_i',
                                refractory=2*ms, reset='v=v_reset_i', method='euler', name='IG')
        app['IG'].v = v_rest_i - 20.*mV

        if (debug):
            app['ISP'] = SpikeMonitor(app['IG'], name='ISP')
            app['ISM'] = StateMonitor(
                app['IG'], ['v'], record=True, name='ISM')
            app['IRM'] = PopulationRateMonitor(app['IG'], name='IRM')

        # poisson generators one-to-all excitatory neurons with plastic connections
        app['S1'] = Synapses(app['PG'], app['EG'], stdp,
                             on_pre=pre, on_post=post, method='euler', name='S1')
        app['S1'].connect()
        app['S1'].w = 'rand()*gmax'  # random weights initialisation
        app['S1'].lr = 1  # enable stdp

        if (debug):
            # some synapses
            app['S1M'] = StateMonitor(
                app['S1'], ['w', 'Apre', 'Apost'], record=app['S1'][380, :4], name='S1M')

        # excitatory neurons one-to-one inhibitory neurons
        app['S2'] = Synapses(app['EG'], app['IG'], 'w : 1',
                             on_pre='ge += w', name='S2')
        app['S2'].connect(j='i')
        app['S2'].delay = 'rand()*10*ms'
        # very strong fixed weights to ensure corresponding inhibitory neuron will always fire
        app['S2'].w = 3

        # inhibitory neurons one-to-all-except-one excitatory neurons
        app['S3'] = Synapses(app['IG'], app['EG'], 'w : 1',
                             on_pre='gi += w', name='S3')
        app['S3'].connect(condition='i!=j')
        app['S3'].delay = 'rand()*5*ms'

        # weights are selected in such a way as to maintain a balance between excitation and ibhibition
        app['S3'].w = .03

        self.net = Network(app.values())
        self.net.run(0*second)
        # self.net.restore(filename='trainall_but_0_epoch_1_eineuron_1000')

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
                self.net['PG'].rates = np.zeros(n_input)*Hz
                self.net.run(0.15*second)

    
    
    # not sure what this function does exactly. I think it makes spike
    # trains out of the test data and saves the data as a python dictionary.
    # However, I'm unclear re: whether this is an evaluation.
    def evaluate(self, X, test_labels):
        self.net['S1'].lr = 0  # stdp off
        store_idx = numpy.zeros(10).astype(int)
        for idx in range(len(X)):

            if (idx % 100 == 0):
                print(str(idx))

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
