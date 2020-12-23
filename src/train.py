from src import config, DQN, ExperienceReplay, EpsilonGreedyStrategy
from tqdm import tqdm
import gym
import torch
from torch import tensor
import torch.nn.functional as F
import argparse

def train_dqn(dqn, experience_replay, optimizer):
    optimizer.zero_grad(set_to_none=True)

    minibatch = experience_replay.sample()
    states = tensor([elem[0] for elem in minibatch], dtype=torch.float32, device=config.DEVICE)
    actions = tensor([elem[1] for elem in minibatch], dtype=torch.int64, device=config.DEVICE)
    rewards = tensor([elem[2] for elem in minibatch], device=config.DEVICE)
    next_states = tensor([elem[3] for elem in minibatch], dtype=torch.float32, device=config.DEVICE)

    preds = dqn(states)
    preds = torch.mul(preds, F.one_hot(actions, config.N_ACTIONS)).sum(axis=1)

    targets = dqn(next_states).detach()
    targets = targets.amax(axis=1)
    targets *= config.DISCOUNT
    targets += rewards
    
    loss = F.mse_loss(preds, targets)
    loss.backward()
    optimizer.step()

def training_loop(model_name=None):
    dqn = DQN().to(config.DEVICE)
    optimizer = torch.optim.RMSprop(dqn.parameters(), lr=config.LEARNING_RATE, momentum=config.GRADIENT_MOMENTUM)
    experience_replay = ExperienceReplay()
    strategy = EpsilonGreedyStrategy()
    env = gym.make(config.ENV_NAME)
    state = env.reset()

    for i in tqdm(range(config.N_FRAMES_TO_TRAIN)):
        if strategy.choose_random():
            action = env.action_space.sample()
        else:
            q_values = dqn(tensor(state, dtype=torch.float32, device=config.DEVICE))
            action = q_values.argmax().item()

        next_state, reward, done, _ = env.step(action)
        experience_replay.append((state, action, reward, next_state))

        if(len(experience_replay) > config.BATCH_SIZE):
            train_dqn(dqn, experience_replay, optimizer)

        if done:
            state = env.reset()
        else:
            state = next_state

        strategy.decrease_epsilon()

    env.close()
    torch.save(dqn.state_dict(), config.MODEL_SAVE_PATH/model_name or config.MODEL_NAME)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", type=str, help="Please add the .pt extension")
    args = parser.parse_args()
    training_loop(model_name=args.model_name)
