from . import config
from random import uniform

class EpsilonGreedyStrategy:
    def __init__(self):
        self.epsilon = 1.0
        self.decay_rate = (1.0 - config.MIN_EPSILON) / (config.N_FRAMES_TO_TRAIN / 10)

    def random_action(self):
        """
        It returns if the action should be random or greedy.

        Return:
        ------
        True: Random Action
        False: Greedy Action
        """
        if uniform(0, 1) > self.epsilon:
            return True
        else:
            return False


    def decrease_epsilon(self):
        self.epsilon -= decay_rate
        self.epsilon = max(config.MIN_EPSILON, self.epsilon)
