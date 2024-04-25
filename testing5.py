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

# 初始化DDPG代理
agents = {agent: DDPGAgent(state_dim=state_size, num_agents=8)
          for agent in env.drones}

done = False
for agent in env.drones:
    print("-----------------")
    print(observations[agent])
while not done:
    actions = {}
    for agent in env.drones:
        action = agents[agent].act(observations[agent])
        actions[agent] = action

    # 只有rewards和done被返回
    rewards, done = env.step(actions)
    env.render()



    # 创建一个空的infos字典
    infos = {agent: {} for agent in env.drones}

    # 更新代理
    for agent in env.drones:
        # 假设next_observations与当前observations相同
        next_observations = observations
        action_patterns = [
            [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1],
            [0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1]
        ]
        action_index = action_patterns.index(action)
        agents[agent].step(observations[agent], action_index, rewards, next_observations[agent], done)

    observations = next_observations

    # 检查所有无人机是否完成
    if all(done for _ in env.drones):  # 假设所有无人机共享相同的完成状态
        done = True
    print(env.time_steps)
    if env.time_steps > env.terminal_time_steps:
        options = {'reset_map': 0}
        observations, infos = env.reset_ddpg(options=options)
        print(observations)
        for agent in env.drones:
            print("-----------------")
            print(observations[agent])
print("训练完成。")
