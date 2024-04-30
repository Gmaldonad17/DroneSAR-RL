import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, OrderedDict

class ResBlock(nn.Module):
    def __init__(self, chan, use_SiLU=True):
        super().__init__()
        activation = nn.SiLU if use_SiLU else nn.ReLU
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            activation(),
            nn.Conv2d(chan, chan, 3, padding=1),
            activation(),
            nn.Conv2d(chan, chan, 1),
        )

    def forward(self, x):
        return self.net(x) + x

# Define the DQN network
class DQN(nn.Module):
    def __init__(self,
                 in_channels=5,
                 num_actions=16,
                 num_layers_resnet=3,
                 ):
        super(DQN, self).__init__()
        self.num_actions = num_actions

        self.init_conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1, stride=1)),
            ('conv2', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2))
        ]))

        resnet_layers = []
        for _ in range(num_layers_resnet):
            resnet_layers.append(ResBlock(chan=16))
        self.resnet_layers = nn.Sequential(
            *resnet_layers
        )

        self.pose_convs = nn.Sequential(OrderedDict([
            ('p_conv1', nn.Conv2d(in_channels=16, out_channels=8, kernel_size=4, stride=2)),
            ('relu1', nn.ReLU()),
            ('p_conv2', nn.Conv2d(in_channels=8, out_channels=1, kernel_size=2, stride=1)),
            ('relu2', nn.ReLU()),
        ]))

        self.out_network = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_actions)
        )

        # self.net = nn.Sequential(
        #     ('init conv', self.init_conv),
        #     ('ResNet', self.resnet_layers),
        #     ('post convs', self.pose_convs),
        #     ('out', self.out_network)
        # )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.resnet_layers(x)
        x = self.pose_convs(x)
        x = self.out_network(x)
        return x

class DQNv0(nn.Module):
    def __init__(self,
                 in_channels=5,
                 num_actions=16):
        super(DQNv0, self).__init__()
        self.num_actions = num_actions
        
        # Initialize convolution layers
        self.init_conv_layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 8, kernel_size=1, stride=1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(8, 16, kernel_size=1, stride=1)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(16, 16, kernel_size=2, stride=2)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(16, 2, kernel_size=2, stride=2)),
            ('relu4', nn.ReLU())
        ]))

        # Initialize fully connected layers
        self.fc_layers = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('fc1', nn.LazyLinear(1024)),
            ('relu4', nn.ReLU()),
            ('fc2', nn.Linear(1024, self.num_actions))
        ]))

    def forward(self, x):
        x = self.init_conv_layers(x)
        x = self.fc_layers(x)
        return x


# Define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

# Define the DQN agent
class DQNAgent:
    def __init__(self, input_shape, num_actions, device):
        self.device = device
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.dqn = DQN(input_shape, num_actions).to(device)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr)
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.dqn(state)
            return q_values.max(1)[1].item()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.dqn(state)
        next_q_values = self.dqn(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = nn.functional.smooth_l1_loss(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)