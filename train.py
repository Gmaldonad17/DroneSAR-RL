import numpy as np
import torch
import argparse
from copy import copy
from custom_environment import landscapev0
from pettingzoo.test import parallel_api_test
from utils import generate_motor_actions, mask_undiscovered_tiles
from configs import config, parse_args

args = parse_args()

def main():
    env = landscapev0(**config.environment)
    observations, infos = env.reset()

    all_actions = generate_motor_actions(max_power_steps=config.max_power_steps)

    done = False

    # Loop until the game is done
    for _ in range(config.episodes):
        done = False
        for _ in range(config.len_eps):
            done = False
            while not done:
                actions = {}
                base_model_input = [env.position_heatmap.astype(np.float32), 
                                    env.discovery_map.astype(np.float32), 
                                    env.features_heatmap.astype(np.float32), 
                                    (mask_undiscovered_tiles\
                                    (env.tile_map, env.discovery_map) / len(env.encyclopedia._biomes.keys())).astype(np.float32)]
                batch = []
                for i, agent in enumerate(env.drones):
                    # Sample random actions for each agent
                    actions[agent] = all_actions[np.random.randint(all_actions.shape[0])]
                    batch.append(base_model_input + [agent.heatmap])
                
                batch = torch.tensor(batch)

                # actions = model(batch)
                # actions = torch.argmax(actions, dim=1)
                # observations, rewards, terminations, truncations, infos = env.step(actions)
                # Render the current state of the environment

                reward, done = env.step(actions)
                print(reward)
                env.render()
                
                if env.time_steps > env.terminal_time_steps:
                    options = {'reset_map': 0}
                    observations, infos = env.reset(options=options)
                    done = True
            
        observations, infos = env.reset()

if __name__ == "__main__":
    main()