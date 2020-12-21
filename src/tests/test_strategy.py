from ..strategy import EpsilonGreedyStrategy
from .. import config

def test_exceed_min_episilon():
    strategy = EpsilonGreedyStrategy()
    for i in range(int(config.N_FRAMES_TO_TRAIN / 10) + 5):
        strategy.decrease_epsilon()
    assert strategy.epsilon == config.MIN_EPSILON