import numpy as np
import torch
import argparse
from tqdm import tqdm
from copy import copy
from custom_environment import landscapev0
from pettingzoo.test import parallel_api_test
from utils import generate_motor_actions, mask_undiscovered_tiles, get_model_input
from configs import config, parse_args
from utils import Metrics
from agents import DQN
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss
from torch.optim.lr_scheduler import StepLR

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

args = parse_args()
# Holds statistics for training
metric_object = Metrics()


def epsilon_greedy(q_values, epsilon=0.2):
    """
    Selects actions using an epsilon-greedy policy.
    
    Args:
    q_values (torch.Tensor): Tensor of shape (batch_size, num_actions) containing the Q-values for each action.
    epsilon (float): Probability of choosing a random action. A higher epsilon increases exploration.

    Returns:
    torch.Tensor: Tensor of shape (batch_size,) containing the indices of the chosen actions.
    """
    batch_size, num_actions = q_values.shape
    random_actions = torch.randint(0, num_actions, (batch_size,)).to(device)  # Random actions for each element in the batch
    best_actions = torch.argmax(q_values, dim=1)  # Best actions according to Q-values

    # Choose whether to explore or exploit for each element in the batch
    choose_random = (torch.rand(batch_size) < epsilon).to(device)  # True with probability epsilon
    actions = torch.where(choose_random, random_actions, best_actions)

    return actions


def get_epsilon(episode, total_episodes, initial_epsilon=1.0, min_epsilon=0.01):
    """
    Calculates epsilon based on the current episode using linear decay.
    
    Args:
    episode (int): Current episode number.
    total_episodes (int): Total number of episodes.
    initial_epsilon (float): Starting value of epsilon.
    min_epsilon (float): Minimum value of epsilon.
    
    Returns:
    float: The calculated epsilon value.
    """
    epsilon = max(min_epsilon, initial_epsilon - (initial_epsilon - min_epsilon) * (episode / total_episodes))
    return epsilon


def main():
    env = landscapev0(**config.environment)
    # observations, infos = env.reset()

    all_actions = generate_motor_actions(max_power_steps=config.max_power_steps)
    model = eval(config.model_name)(
        config.model_config.in_channels, 
        len(all_actions)).to(device)
    
    # Optimizer and scheduler setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.99)
    
    loss_func = smooth_l1_loss

    done = False

    # Loop until the game is done
    pbar = tqdm(range(config.episodes))

    initial_epsilon = 1.0  # Starting with full exploration
    min_epsilon = 0.01    # Minimum exploration

    for episode in pbar:
        current_epsilon = get_epsilon(episode, config.episodes, initial_epsilon, min_epsilon)
        observations, infos = env.reset()
        done = False
        for iter in range(config.len_eps):
            while not done:
                actions = {}
                batch = []
                base_model_input = get_model_input(env)
                for i, agent in enumerate(env.drones):
                    batch.append(base_model_input + [agent.heatmap])
                
                batch = torch.tensor(np.array(batch)).to(device)
                q_values = model(batch)

                actions_taken = epsilon_greedy(q_values, epsilon=current_epsilon)
                for action, drone in zip(actions_taken, env.drones):
                    actions[drone] = all_actions[action]
                
                reward, done = env.step(actions)
                
                if not done:
                    batch_next = []
                    next_state_base = get_model_input(env)
                    for i, agent in enumerate(env.drones):
                        batch_next.append(next_state_base + [agent.heatmap])
                    batch_next = torch.tensor(np.array(batch_next)).to(device)

                    next_q_values = model(batch_next)
                    max_next_q_values = torch.max(next_q_values, dim=1)[0]
                    expected_q_value = reward + config.gamma * max_next_q_values.unsqueeze(1) * (1 - int(done))

                    loss = loss_func(q_values.gather(1, actions_taken.unsqueeze(1)), expected_q_value.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                env.render()
                metric_object.FirstClueFind(env)

                if done:
                    options = {'reset_map': 0}
                    _, _ = env.reset(options=options)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"LR: {current_lr:.6f}, Epsilon: {current_epsilon:.4f}")
            done = False

        metric_object.UpdateMetrics(env)
        _, _ = env.reset()
        done = False 

if __name__ == "__main__":
    main()