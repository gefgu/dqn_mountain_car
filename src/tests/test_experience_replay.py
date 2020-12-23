from src import ExperienceReplay, experience_replay
from numpy import array_equal


def test_sample_minibatch():
    experience_replay = ExperienceReplay()
    for i in range(2):
        experience_replay.append(([1, 1], 2, 3, [4, 4]))
    minibatch = experience_replay.sample(size=2)
    assert [elem[0] for elem in minibatch] == [[1, 1], [1, 1]]
    assert [elem[1] for elem in minibatch] == [2, 2]
    assert [elem[2] for elem in minibatch] == [3, 3]
    assert [elem[3] for elem in minibatch] == [[4, 4], [4, 4]]
