import torch
import torch.nn as nn
from collections import deque, namedtuple
import numpy as np
from utils import rgb_to_gray_tensor
import random

class QNet(torch.nn.Module):
    def __init__(self, n_actions, n_hid=64):
        super(QNet, self).__init__()
        
        self.flatten = nn.Flatten()
        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=8, stride=2),
            # torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=2),
        )
        # [1, 1, 117, 125]
        self.fc = nn.Sequential(
            nn.Linear(117 * 125, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_actions),
        )
    
    def preprocessor(self, observation):
        gray_image = rgb_to_gray_tensor(observation)
        feat_map = self.conv(gray_image)
        return feat_map

    def forward(self, x):
        x = self.preprocessor(x)
        x = self.flatten(x)  # [B, 1 * 117 * 125]
        x = self.fc(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        # TODO: Initialize the buffer
        self.buffer = deque(maxlen=capacity)  # Double-Ended Queue
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])

    # TODO: Implement the add method
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) == self.buffer.maxlen-1:
          print('buffer full!')
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
        # if len(self.buffer) == 100000:
        #   assert False, 'buffer full!'

    # TODO: Implement the sample method
    def sample(self, batch_size):

      batch_size = min(batch_size, len(self.buffer))
      experiences = random.sample(self.buffer, batch_size)

      return {
        'states':      torch.tensor(np.array([e.state for e in experiences])),
        'actions':     torch.tensor(np.array([e.action for e in experiences])),
        'rewards':     torch.tensor(np.array([e.reward for e in experiences])),
        'next_states': torch.tensor(np.array([e.next_state for e in experiences])),
        'dones':       torch.tensor(np.array([e.done for e in experiences]))
      }
    # states [b, 240, 256, 3]

class DQNAgent:
    def __init__(self, action_size):
        # TODO: Initialize some parameters, networks, optimizer, replay buffer, etc.
        self.target_net = QNet(action_size)
        self.train_net = QNet(action_size)
        self.buffer = ReplayBuffer(capacity=250000)
        self.optim = torch.optim.Adam(
            self.train_net.parameters()
        )
        self.tau = 0.005
        self.gamma = 0.99
        self.loss = torch.nn.MSELoss()
        # torch.nn.L1Loss to MSE

    def get_action(self, state, epsilon=None):
        # TODO: Implement the action selection
        # from train net
        # print('state', state.shape)
        if not torch.is_tensor(state):
          state = torch.from_numpy(state.copy())

        if epsilon is not None and random.random() < epsilon:
          act = random.choice([0, 1, 2, 3])
        else:
          act_distr = self.train_net(state)
          act = torch.argmax(act_distr).item()

        return act


    def update(self, soft=True):
        # soft: θ− ← τθ− + (1− τ)θ

        if soft:
            for target_param, train_param in zip(self.target_net.parameters(), self.train_net.parameters()):
                target_param.data.copy_(
                    self.tau * target_param.data + (1.0 - self.tau) * train_param.data
                )
        else:
            # Hard update: copy all parameters
            self.target_net.load_state_dict(self.train_net.state_dict())


    def train(self, episode, episode_loss, step):
        # TODO: Sample a batch from the replay buffer
        data = self.buffer.sample(batch_size=64)

        # TODO: Compute loss and update the model
        # preds to update: Q(st, at)

        actions_taken = torch.tensor(data['actions'], dtype=torch.long)
        action_distrs = self.train_net(data['states'])
        preds = action_distrs[torch.arange(action_distrs.size(0)), actions_taken]

        # target from the target net
        max_next_Qs = self.target_net(data['next_states']).max(dim=1).values
        max_next_Qs = max_next_Qs * (~data['dones'])
        targets = data['rewards'] + self.gamma * max_next_Qs
        loss = self.loss(preds, targets.to(torch.float))
        # init priorities w the batch of TD error

        episode_loss.append(loss.item())

        self.optim.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.train_net.parameters(), 100)
        self.optim.step()

        # TODO: Update target network periodically
        # if step % 5 == 0:
        self.update(soft=True)
