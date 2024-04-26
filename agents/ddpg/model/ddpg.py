import tensorflow as tf
from keras import layers, Model

class Actor(Model):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(8, activation='softmax')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.out(x)


class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.state_fc = layers.Dense(128, activation='relu')
        self.action_fc = layers.Dense(128, activation='relu')
        self.concat = layers.Concatenate()
        self.fc1 = layers.Dense(256, activation='relu')
        self.out = layers.Dense(1)  # Value of the state-action pair

    def call(self, inputs):
        state, action = inputs
        state_out = self.state_fc(state)
        action_out = self.action_fc(action)
        concat = self.concat([state_out, action_out])
        x = self.fc1(concat)
        return self.out(x)

