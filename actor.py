
import torch
from unused.DQN_lunar import MarioNet
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
import torch.nn.functional as F

import torch
import random
from collections import deque
import numpy as np
from model import DQN

class Mario:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.update_target_every = 1000
        self.step_counter = 0

    def act(self, state, epsgreedy):
        if epsgreedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state = state.unsqueeze(0).float().to(self.device)
            q_values = self.net(state)
            return q_values.argmax().item()

    def save(self, transition):
        self.replay_buffer.append(transition)

    def sample(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states).float().to(self.device),
                torch.tensor(actions).long().to(self.device),
                torch.tensor(rewards).float().to(self.device),
                torch.stack(next_states).float().to(self.device),
                torch.tensor(dones).float().to(self.device))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample()

        q_values = self.net(states)[range(self.batch_size), actions]
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.smooth_l1_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)



    


    