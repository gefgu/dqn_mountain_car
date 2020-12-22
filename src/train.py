from torch._C import dtype
import config
from dqn import DQN
from experience_replay import ExperienceReplay
from strategy import EpsilonGreedyStrategy
from tqdm import tqdm
import gym
import torch
from torch import tensor
import numpy as np
import torch.nn.functional as F
import sys

def train_dqn(dqn, experience_replay, optimizer):
    optimizer.zero_grad()

    minibatch = experience_replay.sample()
    states = tensor([elem[0] for elem in minibatch], dtype=torch.float32)
    actions = tensor([elem[1] for elem in minibatch], dtype=torch.int64)
    rewards = tensor([elem[2] for elem in minibatch])
    next_states = tensor([elem[3] for elem in minibatch], dtype=torch.float32)

    preds = dqn(states)
    preds = torch.mul(preds, F.one_hot(actions, config.N_ACTIONS)).sum(axis=1)

    targets = dqn(next_states).detach()
    targets = targets.amax(axis=1)
    targets *= config.DISCOUNT
    targets += rewards
    
    loss = F.mse_loss(preds, targets)
    loss.backward()
    optimizer.step()




def training_loop():
    dqn = DQN()
    optimizer = torch.optim.RMSprop(dqn.parameters(), lr=config.LEARNING_RATE)
    experience_replay = ExperienceReplay()
    strategy = EpsilonGreedyStrategy()
    env = gym.make(config.ENV_NAME)
    state = env.reset()

    for i in tqdm(range(config.N_FRAMES_TO_TRAIN)):

        if strategy.random_action():
            action = env.action_space.sample()
        else:
            q_values = dqn(torch.from_numpy(state.astype(np.float32)))
            action = q_values.argmax().item()

        next_state, reward, done, _ = env.step(action)
        experience_replay.append((state, action, reward, next_state))

        if(len(experience_replay) > config.BATCH_SIZE):
            train_dqn(dqn, experience_replay, optimizer)

        if done:
            state = env.reset()
        else:
            state = next_state
    env.close()
    torch.save(dqn.state_dict(), config.MODEL_SAVE_PATH/"model.pt")




if __name__ == "__main__":
    training_loop()