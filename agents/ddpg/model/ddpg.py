import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


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

class Actor(nn.Module):     #input: state; output: action
    def __init__(self,
                 in_channels=5,
                 num_layers_resnet=3,
                 num_actions=16
                 ):
        super(Actor, self).__init__()
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=2, stride=1)

        resnet_layers = []
        for _ in range(num_layers_resnet):
            resnet_layers.append(ResBlock(chan=64))
        self.resnet_layers = nn.Sequential(
            *resnet_layers
        )

        self.pose_convs = nn.Sequential(
            ('p_conv1', nn.Conv2d(in_channels=64, out_channels=16, kernel_size=2, stride=1)),
            ('relu1', nn.ReLU()),
            ('p_conv2', nn.Conv2d(in_channels=8, out_channels=1, kernel_size=2, stride=1)),
            ('relu2', nn.ReLU()),
        )

        self.out_network = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_actions)
        )

        self.net = nn.Sequential(
            ('init conv', self.init_conv),
            ('ResNet', self.resnet_layers),
            ('post convs', self.pose_convs),
            ('out', self.out_network)
        )

    def forward(self, inputs):
        return self.net(inputs)

class Critic(nn.Module):        #input: state, output of actor(action) ; output: Q(s,a)
    def __init__(self):
        super(Critic, self).__init__()
        self.state_fc = nn.Linear(128, 128)
        self.action_fc = nn.Linear(128, 128) 
        self.fc1 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, inputs):
        state, action = inputs
        state_out = F.relu(self.state_fc(state))
        action_out = F.relu(self.action_fc(action))
        concat = torch.cat([state_out, action_out], dim=1)
        x = F.relu(self.fc1(concat))
        x = self.out(x)
        return x

