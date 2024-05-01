import numpy as np
import torch
import cv2
import pygame
from custom_environment import landscapev0
from utils import generate_motor_actions, get_model_input
from configs import config
from agents import DQN, DQNv0

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_model(model_path):
    model = eval(config.model_name)(
        config.model_config.in_channels, 
        len(generate_motor_actions(max_power_steps=config.max_power_steps))).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main():
    env = landscapev0(**config.environment)
    observations, infos = env.reset()

    all_actions = generate_motor_actions(max_power_steps=config.max_power_steps)
    model = load_model('/saved_models/model.pth')


    done = False
    while not done:
        actions = {}
        batch = []
        base_model_input = get_model_input(env)
        for i, agent in enumerate(env.drones):
            heatmap = agent.heatmap + (i / (len(env.drones) + 4))
            batch.append(base_model_input + [heatmap])

        batch = torch.tensor(np.array(batch)).to(device)
        with torch.no_grad():
            q_values = model(batch)

        actions_taken = torch.argmax(q_values, dim=1)
        for action, drone in zip(actions_taken, env.drones):
            actions[drone] = all_actions[action]

        _, done, _ = env.step(actions)

        env.render()



if __name__ == "__main__":
    main()