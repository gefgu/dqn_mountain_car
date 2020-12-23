from src import DQN, config
import gym 
import torch
import numpy as np

def show_model_playing():
    env = gym.make(config.ENV_NAME)
    state = env.reset()

    dqn = DQN().to(config.DEVICE)
    dqn.load_state_dict(torch.load(config.MODEL_SAVE_PATH/config.MODEL_NAME))


    for i in range(config.N_STEPS_TO_PLAY):
        env.render()
        q_values = dqn(torch.from_numpy(state.astype(np.float32)))
        action = q_values.argmax().item()

        next_state, _, done, _ = env.step(action)

        if done:
            state = env.reset()
        else:
            state = next_state

    env.close()


if __name__ == "__main__":
    show_model_playing()