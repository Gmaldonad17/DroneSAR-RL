

#for prison_game()



from agents import DDPGAgent
import numpy as np
from pettingzoo.butterfly import cooperative_pong_v5
from custom_environment import prison_game, landscapev0
from pettingzoo.test import parallel_api_test


env = prison_game()
observations, infos = env.reset()
# 初始化环境

observation_space = env.observation_space('prisoner')
action_space = env.action_space('prisoner')

# 初始化DDPG代理
agent = DDPGAgent(state_dim=observation_space.shape[0], action_dim=action_space.n,num_agents=8)

# 训练过程
for episode in range(1000):  # 运行1000个episode
    observations,_ = env.reset()
    done = False
    total_reward = 0

    while not done:
        actions = {}
        for agent_id, obs in observations.items():
            action = agent.act(obs)  # 让DDPG代理根据观测选择动作
            actions[agent_id] = action

        next_observations, rewards, dones, infos = env.step(actions)

        for agent_id in observations:
            agent.remember(observations[agent_id], actions[agent_id], rewards[agent_id], next_observations[agent_id],
                           dones[agent_id])
            agent.update()  # 更新DDPG网络

        observations = next_observations
        done = all(dones.values())
        total_reward += sum(rewards.values())

    print(f'Episode {episode}: Total Reward = {total_reward}')
