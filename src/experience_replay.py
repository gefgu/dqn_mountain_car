from collections import deque 
from random import sample
from numpy import array

class ExperienceReplay:
    def __init__(self, maxlen=1000000):
        self.memory = deque(maxlen=maxlen)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_minibatch(self, bs=32):
        minibatch = sample(self.memory, k=bs)
        minibatch = array(minibatch)
        return minibatch[:, 0], minibatch[:, 1], minibatch[:, 2], \
                minibatch[:, 3], minibatch[:, 4]