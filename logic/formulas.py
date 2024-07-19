from constants import *

class FORMULAS():

    # stdp formula
    stdp = '''w : 1
        lr : 1 (shared)
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
    '''
    
    # pre
    pre = '''ge += w
        Apre += dApre
        w = clip(w + lr*Apost, 0, gmax)
    '''
    
    # post
    post = '''Apost += dApost
        w = clip(w + lr*Apre, 0, gmax)
    '''

    # excitatory neuron formula:
    neuron_e = '''
        dv/dt = (ge*(0*mV-v) + gi*(-100*mV-v) + (v_rest_e-v)) / (100*ms) : volt
        dge/dt = -ge / (5*ms) : 1
        dgi/dt = -gi / (10*ms) : 1
    '''

    # ibhibitory neuron formula:
    neuron_i = '''
        dv/dt = (ge*(0*mV-v) + (v_rest_i-v)) / (10*ms) : volt
        dge/dt = -ge / (5*ms) : 1
    '''