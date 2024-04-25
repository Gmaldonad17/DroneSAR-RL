

#for landscapev0(),with next_observations updated by steps




from agents import DDPGAgent
import numpy as np
from pettingzoo.butterfly import cooperative_pong_v5
from custom_environment import prison_game, landscapev0
from pettingzoo.test import parallel_api_test
# 初始化环境
env = landscapev0()
observations, infos = env.reset_ddpg()

# 假设我们知道状态和动作的大小，如果不确定，需要从环境中获取
state_size = observations[env.drones[0]].shape[0]  # 假设所有无人机的观测空间相同
action_size = 2

# 初始化DDPG代理
agents = {agent: DDPGAgent(state_dim=state_size,num_agents=8)
          for agent in env.drones}

done = False
while not done:
    actions = {}
    for agent in env.drones:
        action = agents[agent].act(observations[agent])
        actions[agent] = action

    next_observations, rewards, dones, infos = env.step(actions)
    env.render()

    # 更新代理
    for agent in env.drones:
        agents[agent].step(observations[agent], actions[agent], rewards[agent], next_observations[agent], dones[agent])

    observations = next_observations

    # 检查所有无人机是否完成
    if all(dones.values()):
        done = True

    # 如果达到终止时间步，重置环境
    if env.time_steps > env.terminal_time_steps:
        observations, infos = env.reset_ddpg(options={'reset_map': 0})

print("训练完成。")
