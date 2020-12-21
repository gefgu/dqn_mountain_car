
from ..dqn import DQN
from .. import config
from torch import rand


def test_dqn_output_shape():
    dqn = DQN()
    out = dqn(rand(config.N_STATE_FEATURES))
    assert out.shape == (config.N_ACTIONS,)
