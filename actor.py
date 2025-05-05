
import torch
# from unused.DQN_lunar import MarioNet
# from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
# from tensordict import TensorDict
import torch.nn.functional as F
from PIL import Image
import torch
import random
from collections import deque
import numpy as np
from model import DQN
from utils import PrioritizedReplayBuffer

class Mario:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.train_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.train_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.train_net.parameters(), lr=3e-4) 
        # self.replay_buffer = deque(maxlen=50000)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)

        self.batch_size = 256
    
        self.gamma = 0.99

        self.check_every = 100
        self.step_counter = 0
        self.tau = 5e-5
        self.n_step = 9
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.1
        
    def get_n_step_transition(self):
        # done or full. either case, if there's a done, it should be at the last transition
        # (state, action, reward, next_state, done) * at most 9
        init_state = self.n_step_buffer[0][0]
        init_action = self.n_step_buffer[0][1]
        next_state = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][-1]
        n_step_rwd = 0
        
        for p, transition in enumerate(self.n_step_buffer):
            rwd = transition[2]
            n_step_rwd += rwd * (self.gamma ** p)
        
        return (init_state, init_action, n_step_rwd, next_state, done)
        
    def n_step_act(self, env, state, frames, epsgreedy):
        
        action = self.act(state, epsgreedy=epsgreedy)
        next_state, reward, done, _ = env.step(action)
        
        # n_step_rwd += reward * (self.gamma ** step)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float) / 255.0
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if done or len(self.n_step_buffer) == self.n_step_buffer.maxlen:
            n_step_transition = self.get_n_step_transition()
            self.save(n_step_transition)
        
        frame = env.render(mode='rgb_array')
        frames.append(Image.fromarray(frame))
        
        return next_state, reward, done
        

    def act(self, state, epsgreedy):
        
        if epsgreedy and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            # print('state', type(state)) # torch.Tensor
            self.train_net.reset_noise()
            state = state.unsqueeze(0).float().to(self.device)
            q_values = self.train_net(state)
            action = q_values.argmax().item()
        
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        return action
        
    def save(self, transition):
        self.replay_buffer.add(transition)
        # if len(self.replay_buffer) == self.replay_buffer.maxlen - 1:
        #     print('buffer full!')
    
    # def save(self, transition):
    #     self.replay_buffer.add(transition)

    
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

        # states, actions, rewards, next_states, dones = self.sample()
        states, actions, rewards, next_states, dones, idxs, weights = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        self.train_net.reset_noise()
        self.target_net.reset_noise()

        q_values = self.train_net(states)[range(self.batch_size), actions]
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.gamma**self.n_step) * max_next_q * (1 - dones)
        
        # loss = F.smooth_l1_loss(q_values, target_q)
        loss = (F.smooth_l1_loss(q_values, target_q, reduction='none') * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities
        td_error = (q_values - target_q).detach().cpu().numpy()
        self.replay_buffer.update_priorities(idxs, np.abs(td_error))

        self.step_counter += 1
        
        # soft update at every step
        for target_param, train_param in zip(self.target_net.parameters(), self.train_net.parameters()):
            target_param.data.copy_(
                self.tau * train_param.data + (1.0 - self.tau) * target_param.data
            )

        
        
        
    # def newupdate(self):
    #     if len(self.replay_buffer) < self.batch_size:
    #         return

    #     states, actions, rewards, next_states, dones, weights, idxs = self.replay_buffer.sample(self.batch_size)
    #     states = states.float().to(self.device)
    #     next_states = next_states.float().to(self.device)
    #     actions = actions.long().to(self.device)
    #     rewards = rewards.float().to(self.device)
    #     dones = dones.float().to(self.device)
    #     weights = weights.to(self.device)

    #     q_values = self.train_net(states)[range(self.batch_size), actions]
    #     with torch.no_grad():
    #         max_next_q = self.target_net(next_states).max(1)[0]
    #         target_q = rewards + (self.gamma**self.n_step) * max_next_q * (1 - dones)

    #     td_errors = target_q - q_values
    #     loss = (weights * F.smooth_l1_loss(q_values, target_q, reduction='none')).mean()

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     self.train_net.reset_noise()
    #     self.replay_buffer.update_priorities(idxs, td_errors.detach())

    #     for target_param, train_param in zip(self.target_net.parameters(), self.train_net.parameters()):
    #         target_param.data.copy_((1.0 - self.tau) * train_param.data + self.tau * target_param.data)

    #     self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        



    


    