from pettingzoo.butterfly import cooperative_pong_v5


from custom_environment import prison_game, landscapev0

from pettingzoo.test import parallel_api_test


env = landscapev0()
observations, infos = env.reset()

# Define a variable to keep track of whether the game is done
done = False

# Loop until the game is done
while not done:
    # actions = {}
    # for agent in env.agents:
    #     # Sample random actions for each agent
    #     action_space = env.action_space(agent)
    #     actions[agent] = action_space.sample()
    
    # Perform a step in the environment using the sampled actions
    # observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Render the current state of the environment
    env.render()
    
    # Check if the game is done (either terminated or truncated)
    # if any(terminations.values()) or any(truncations.values()):
        # done = True