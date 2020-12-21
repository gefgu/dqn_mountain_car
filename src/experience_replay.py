from collections import deque 
from random import sample
from numpy import array
from torch._C import dtype
import config

class ExperienceReplay:
    def __init__(self, maxlen=1000000):
        self.memory = deque(maxlen=maxlen)
    
    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def size(self):
        return len(self.memory)

    def sample_minibatch(self, bs=config.BATCH_SIZE):
        minibatch = sample(self.memory, k=bs)
        minibatch = array(minibatch, dtype=object)
        return minibatch[:, 0], minibatch[:, 1], minibatch[:, 2], minibatch[:, 3]