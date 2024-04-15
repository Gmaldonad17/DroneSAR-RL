import numpy as np
from pettingzoo.butterfly import cooperative_pong_v5

from custom_environment import prison_game, landscapev0

from pettingzoo.test import parallel_api_test


env = landscapev0()
observations, infos = env.reset()

# parallel_api_test(env, num_cycles=1_000_000)


done = False

# Loop until the game is done
all_actions = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1],
                        [0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 0, 1]])
actions = {}
while not done:
    # actions = {}
    for i, agent in enumerate(env.drones):
        # Sample random actions for each agent
        actions[agent] = all_actions[i]
        # action_space = env.action_space(agent)
        # actions[agent] = action_space.sample()
    
    # Perform a step in the environment using the sampled actions
    # observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Render the current state of the environment
    env.step(actions)
    env.render()
    
    # Check if the game is done (either terminated or truncated)
    # if any(terminations.values()) or any(truncations.values()):
        # done = True