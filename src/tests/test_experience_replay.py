from ..experience_replay import ExperienceReplay
from numpy import array
from numpy import array_equal


def test_sample_minibatch():
    experience_replay = ExperienceReplay(maxlen=3)
    for i in range(2):
        experience_replay.store_transition(1, 2, 3, 4, False)
    states, actions, rewards, next_states, dones = experience_replay.sample_minibatch(bs=2)
    assert array_equal(states, [1, 1])
    assert array_equal(actions, [2, 2])
    assert array_equal(rewards, [3, 3])
    assert array_equal(next_states, [4, 4])
    assert array_equal(dones, [0, 0])