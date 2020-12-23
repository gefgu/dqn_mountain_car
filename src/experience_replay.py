import random
from src import config

class ExperienceReplay:
    """It's like a Ring Buffer"""
    def __init__(self, size=config.EXPERIENCE_SIZE):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
    
    def append(self, transition):
        """
        Params
        ------
        transition: Tuple -> (state, action, reward, next_state)
        """

        self.data[self.end] = transition
        self.end = (self.end + 1) % len(self.data)
        
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start 
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def sample(self, size=config.BATCH_SIZE):
        idxs = random.sample(range(len(self)), k=size)
        return [self[i] for i in idxs]