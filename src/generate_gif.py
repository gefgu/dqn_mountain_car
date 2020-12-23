from src import DQN, config, ExperienceReplay
import gym 
import torch
import torch 
from torch import tensor
import argparse
from moviepy.editor import ImageSequenceClip

def generate_gif(model_name=None, gif_name=None):
    env = gym.make(config.ENV_NAME)
    state = env.reset()

    dqn = DQN().to(config.DEVICE)
    dqn.load_state_dict(torch.load(config.MODEL_SAVE_PATH/model_name or config.MODEL_NAME))

    frames_array = ExperienceReplay(size=1000)

    for i in range(1000):
        frames_array.append(env.render(mode="rgb_array"))
        q_values = dqn(tensor(state, dtype=torch.float32, device=config.DEVICE))
        action = q_values.argmax().item()

        state, _, done, _ = env.step(action)
        if done:
            break

    env.close()

    clip = ImageSequenceClip(list(frames_array), fps=30)
    clip.write_gif(config.GIF_SAVE_PATH/(gif_name or config.GIF_NAME))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", type=str, help="Please add the .pt extension")
    parser.add_argument("-gif_name", type=str, help="Please add the .gif extension")
    args = parser.parse_args()
    generate_gif(model_name=args.model_name, gif_name=args.gif_name)