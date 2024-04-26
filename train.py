import numpy as np
import torch
import argparse
from copy import copy
from custom_environment import landscapev0
from pettingzoo.test import parallel_api_test
from utils import generate_motor_actions, mask_undiscovered_tiles, get_model_input
from configs import config, parse_args
from utils import Metrics
from agents import DQN
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

args = parse_args()
# Holds statistics for training
metric_object = Metrics()

def main():
    env = landscapev0(**config.environment)
    observations, infos = env.reset()

    all_actions = generate_motor_actions(max_power_steps=config.max_power_steps)
    model = eval(config.model_name)(
        config.model_config.in_channels, 
        len(all_actions)).to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    
    loss_func = eval(config.loss_function)

    done = False

    # Loop until the game is done
    for _ in range(config.episodes):
        for _ in range(config.len_eps):
            while not done:
                actions = {}
                # Inputs shared between all drones
                
                batch = []
                base_model_input = get_model_input(env)
                # Add each drones input to a list
                for i, agent in enumerate(env.drones):
                    batch.append(base_model_input + [agent.heatmap])
                
                batch = torch.tensor(np.array(batch)).to(device)
                q_values = model(batch)

                for action, drone in zip(q_values, env.drones):
                    actions[drone] = all_actions[np.argmax(action.cpu().detach().numpy())]

                reward, done = env.step(actions)

                batch = []
                next_state_base = get_model_input(env)
                for i, agent in enumerate(env.drones):
                    batch.append(base_model_input + [agent.heatmap])
                batch = torch.tensor(np.array(batch)).to(device)

                next_q_value = model(batch)
                expected_q_value = reward + config.gamma * next_q_value * (1 - done)

                loss = loss_func(q_values, expected_q_value.detach())
                print(reward)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                env.render()
                # Check if the first clue has been found
                metric_object.FirstClueFind(env)
                # If the simulation times out, reset
                if done:
                    options = {'reset_map': 0}
                    _, _ = env.reset(options=options)
        
            done = False

        # Save metrics for run and reset variables, reset the environment
        metric_object.UpdateMetrics(env)
        _, _ = env.reset()
        done = False
        

    # Plot graphs after all episodes are done
    metric_object.GraphResults()    

if __name__ == "__main__":
    main()