from torch._C import dtype
import config
from dqn import DQN
from experience_replay import ExperienceReplay
from strategy import EpsilonGreedyStrategy
from tqdm import tqdm
import gym
import torch
import numpy as np
import torch.nn.functional as F

def train_dqn(dqn, experience_replay, optimizer):
    optimizer.zero_grad()

    states, actions, rewards, next_states = experience_replay.sample_minibatch()

    states = np.stack(states).astype(np.float32)
    actions = torch.from_numpy(np.stack(actions).astype(np.int64))
    rewards = torch.from_numpy(np.stack(rewards))
    next_states = np.stack(next_states).astype(np.float32)


    preds = dqn(torch.from_numpy(states))
    preds = torch.mul(preds, F.one_hot(actions, config.N_ACTIONS)).sum(axis=1)

    targets = dqn(torch.from_numpy(next_states)).detach()
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

        next_state, reward, done, info = env.step(action)
        experience_replay.store_transition(state, action, reward, next_state)

        if(experience_replay.size() > config.BATCH_SIZE):
            train_dqn(dqn, experience_replay, optimizer)

        if done:
            state = env.reset()
        else:
            state = next_state




if __name__ == "__main__":
    training_loop()