from .model.ddpg import Actor
from .model.ddpg import Critic

import tensorflow as tf
from keras import layers, Model
import numpy as np
from collections import deque
import random
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np


class DDPGAgent:
    def __init__(self, state_dim, num_agents, actor_lr=0.001, critic_lr=0.002, gamma=0.99, tau=0.005,
                 buffer_size=100000, batch_size=64):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.actor = Actor()
        self.critic = Critic()
        self.target_actor = Actor()
        self.target_critic = Critic()

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr), loss='mse')
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr), loss='mse')

        self.update_target_network(tau=1)

    def update_target_network(self, tau=None):
        if tau is None:
            tau = self.tau
        new_actor_weights = [(1 - tau) * taw + tau * aw for taw, aw in
                             zip(self.target_actor.get_weights(), self.actor.get_weights())]
        new_critic_weights = [(1 - tau) * tcw + tau * cw for tcw, cw in
                              zip(self.target_critic.get_weights(), self.critic.get_weights())]

        self.target_actor.set_weights(new_actor_weights)
        self.target_critic.set_weights(new_critic_weights)

    def act(self, state):
        action_patterns = [
            [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1],
            [0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1]
        ]

        state = np.expand_dims(state, axis=0)
        print("shape of state:",state.shape)
        action_probs = self.actor(state)  # Actor网络输出八种动作模式的概率
        action_index = np.argmax(action_probs.numpy())  # 找到概率最高的索引
        return action_patterns[action_index]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            # 以下是一个简化的DQN学习过程，实际上应根据DDPG算法进行调整
            target = reward
            if not done:
                target += self.gamma * np.amax(self.critic([next_state, self.actor(next_state)]).numpy())
            target_f = self.critic([state, action]).numpy()
            target_f[0][0] = target
            self.critic.fit([state, action], target_f, epochs=1, verbose=0)

    def step(self, state, action, reward, next_state, done):
        # 将经验保存到回放缓冲区
        self.memory.append((state, action, reward, next_state, done))

        # 检查是否已经积累足够的经验进行学习
        '''if len(self.memory) > self.batch_size:
            self.learn()'''

    def learn(self):
        minibatch = random.sample(self.memory, self.batch_size)

        # 为简化，这里使用 list 解包
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # 转换为合适的 numpy 格式
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=bool)

        # 计算目标 Q 值
        not_dones = ~dones
        next_actions = self.target_actor.predict(next_states)
        next_Q_values = self.target_critic.predict([next_states, next_actions])
        target_Q_values = rewards + self.gamma * next_Q_values * not_dones

        # 更新 Critic 网络
        critic_loss = self.critic.train_on_batch([states, actions], target_Q_values)

        # 更新 Actor 网络：这里使用梯度翻转技术（需要自定义训练循环）
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            Q_values = self.critic([states, actions], training=True)
            actor_loss = -tf.reduce_mean(Q_values)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # 软更新目标网络
        self.update_target_network()


'''
    def train(self, env, episodes=1000):
        for episode in range(episodes):
            observations = env.reset()
            done = {agent_id: False for agent_id in observations}
            total_reward = 0

            while not all(done.values()):
                actions = {agent_id: self.act(obs) for agent_id, obs in observations.items()}

                next_observations, rewards, dones, infos = env.step(actions)

                for agent_id in observations:
                    self.remember(observations[agent_id], actions[agent_id], rewards[agent_id],
                                  next_observations[agent_id], dones[agent_id])

                self.replay()
                self.update_target_network()

                observations = next_observations
                done = dones
                total_reward += sum(rewards.values())

            print(f'Episode {episode}: Total Reward = {total_reward}')

'''