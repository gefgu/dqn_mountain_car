from src import DQN, config
import gym 
import torch
import torch 
from torch import tensor
import argparse

def show_model(model_name=None):
    env = gym.make(config.ENV_NAME)
    state = env.reset()

    dqn = DQN().to(config.DEVICE)
    dqn.load_state_dict(torch.load(config.MODEL_SAVE_PATH/model_name or config.MODEL_NAME))


    for i in range(config.N_STEPS_TO_PLAY):
        env.render()
        q_values = dqn(tensor(state, dtype=torch.float32, device=config.DEVICE))
        action = q_values.argmax().item()

        next_state, _, done, _ = env.step(action)

        if done:
            state = env.reset()
        else:
            state = next_state

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", type=str, help="Please add the .pt extension")
    args = parser.parse_args()
    show_model(model_name=args.model_name)