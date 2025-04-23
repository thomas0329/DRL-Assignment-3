#!/usr/bin/env python
# coding: utf-8

# # Deep Reinforcement Learning Class Spring 2025 Assignment 3
# 
# This is the first part of the assignment. In this part, you will learn DQN and its variants with the LunarLander environment from OpenAI Gym.
# 
# You need to fill in the missing code snippets (marked by TODO). Feel free to modify the code structure based on your understanding, but **you are forbidden to use any external RL libraries like Stable Baselines3**, **RLlib**, etc.
# 
# Make a copy of this notebook using File > Save a copy in Drive and edit it with your answers.
# 
# WARNING: Do not put your name or any other personal identification information in this notebook.
# 
# **[04/17 10:48] Update:** Test 10 episodes -> Test **100** episodes.

# ## Quention 1: LunarLander and Random Agent (5%)
# 
# We will start by loading the LunarLander environment and creating a random agent.

# First, install the required packages and import the necessary libraries.

# In[16]:


# get_ipython().system('pip install swig')
# get_ipython().system('pip install gymnasium[box2d]')


# In[1]:


import gymnasium as gym
import os
import matplotlib.pyplot as plt
import imageio
import numpy as np
from IPython.display import Image
import torch
import random
from collections import deque


# ### Understanding the LunarLander Environment
# 
# The LunarLander environment simulates a 2D lunar landing task, where an agent (a spacecraft) must learn to land safely on a designated landing pad. The lander is controlled using discrete actions that fire its engines in different directions
# 
# More information about the environment can be found [here](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

# In[2]:


env = gym.make("LunarLander-v3", render_mode="rgb_array")
action_size = env.action_space.n
# print("action size: ", action_size)
state_size = env.observation_space.shape[0]
# print("state size: ", state_size)


# ### Implementing the Random Agent (5%)
# 
# We will implement a random agent that selects actions randomly from the action space of the environment. Random agents are useful for testing the environment and ensuring that it is working as expected.
# 

# In[ ]:


# frames = [] # Store the frames for the animation

# observation, _ = env.reset()
# done = False
# total_reward = 0
# while not done:
#     # TODO: Select an action randomly
#     action = env.action_space.sample()
#     # TODO: Take a step in the environment
#     observation, reward, terminated, truncated, info = env.step(action)

#     # TODO: Check if the episode is done
#     done = terminated or truncated
#     # TODO: Update the total reward
#     total_reward += reward

#     frames.append(env.render()) # Save the frame for the animation

# env.close()
# print("Total reward:", total_reward)

# # Visulization
# ### Do not modify the follwing code—any changes will result in a score of 0 for this question.
# gif_path = "random_agent.gif"
# imageio.mimsave(gif_path, frames, fps=30)
# Image(filename=gif_path)


# ## Question 2: DQN Agent (15%)
# 
# Next, we will implement a Deep Q-Networks (DQN) agent that learns to play the LunarLander-v3 game. DQN is a reinforcement learning algorithm that uses a deep neural network to approximate the Q-function, allowing the agent to estimate the expected future rewards for each action and learn an optimal policy.
# 
# In this question, we will:
# 
# 1. Define a neural network model to approximate the Q-function.
# 2. Implement the experience replay buffer to store past experiences.
# 3. Implement the DQN agent that interacts with the environment, updates the network, and learns to play the game.
# 
# By completing this section, you will build the core components of a DQN-based reinforcement learning agent that can successfully land the spacecraft in the LunarLander-v3 environment! 🚀

# ### Defining the Neural Network Model
# 

# In[3]:


import torch.nn as nn
class QNet(torch.nn.Module):
    def __init__(self, n_states, n_actions, n_hid=64):
        super(QNet, self).__init__()
        # TODO: Define the neural network

        self.fc = nn.Sequential(
            nn.Linear(n_states, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_actions),
        )

    def forward(self, x):
        return self.fc(x)


# ### Implementing the Experience Replay Buffer

# In[4]:


from collections import deque, namedtuple

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


# ### Defining the DQN Agent

# In[5]:


from itertools import chain

class DQNAgent:
    def __init__(self, state_size, action_size):
        # TODO: Initialize some parameters, networks, optimizer, replay buffer, etc.
        self.target_net = QNet(state_size, action_size)
        self.train_net = QNet(state_size, action_size)
        self.buffer = ReplayBuffer(capacity=250000)
        self.optim = torch.optim.Adam(
            self.train_net.parameters()
        )
        self.tau = 0.005
        self.gamma = 0.99
        self.loss = torch.nn.MSELoss()

    def get_action(self, state, epsilon=None):
        # TODO: Implement the action selection
        # from train net
        if not torch.is_tensor(state):
          state = torch.from_numpy(state)

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


# ### Setting Up the Training Loop

# In[ ]:


# agent = DQNAgent(state_size, action_size)
# # TODO: Determine the number of episodes for training
# num_episodes = 2500

# reward_history = [] # Store the total rewards for each episode
# eps_start = 1.0
# eps_end = 0.1
# eps_decay = 0.995
# epsilon = eps_start
# for episode in range(num_episodes):

#     # TODO: Reset the environment
#     observation, info = env.reset()
#     # print('observation', observation)
#     total_reward = 0
#     done = False
#     episode_loss = []
#     step = 0
#     while not done:
#         # TODO: Select an action from the agent
#         action = agent.get_action(observation, epsilon)
#         next_observation, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         # TODO: Add the experience to the replay buffer and train the agent
#         agent.buffer.add(observation, action, reward, next_observation, done)

#         agent.train(episode, episode_loss, step)

#         # TODO: Update the state and total reward
#         total_reward += reward
#         observation = next_observation
#         step += 1

#     # print(f"Episode {episode}, Reward: {total_reward}, avg_loss: {sum(episode_loss) / len(episode_loss)}")
#     reward_history.append(total_reward)

#     if episode % 100 == 0:
#       last_100_avg_rwd = sum(reward_history[-100:]) / len(reward_history[-100:])
#       print(f"Episode {episode}, Avg of last 100 episodes: {last_100_avg_rwd}")
#       if last_100_avg_rwd > 250:
#         print('saving weights')
#         torch.save(agent.train_net.state_dict(), 'train_net.pth')
#       if last_100_avg_rwd > 260:
#         print("Early stopping")
#         break
#     epsilon = max(eps_end, epsilon*eps_decay)


# Plot the reward history to see how the agent's performance changes over time!
# 

# In[ ]:


# plt.plot(reward_history)
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.title("Training History")
# plt.show()


# ### Testing the DQN Agent (15%)
# 
# Test your DQN agent for 100 episodes and calculate the average reward.
# 
# You will get full points if the average reward is greater than 250.
# 
# Do not modify the cell below, or you will get zero points for this question.

# In[6]:


# agent = DQNAgent(state_size, action_size)
# agent.train_net.load_state_dict(torch.load('train_net.pth'))
# # Average reward: 251.96964761419397


# # In[ ]:


# # frames = [] # Uncomment to store the frames for the animation
# reward_history = []
# for _ in range(100):

#     state, _ = env.reset()
#     total_reward = 0
#     done = False

#     while not done:
#         action = agent.get_action(state)
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated

#         total_reward += reward
#         state = next_state
#         # frames.append(env.render()) # Uncomment to store the frames for the animation

#     print(f"Reward: {total_reward}")
#     reward_history.append(total_reward)

# env.close()

# print("Average reward:", np.mean(reward_history))

# Uncomment to show the animation
# gif_path = "DQN.gif"
# imageio.mimsave(gif_path, frames, fps=30)
# Image(filename=gif_path)


# ## Question 3: Improving the DQN Agent (15%)
# 
# In this question, you need to improve the DQN agent by implementing one or more of the following DQN variants:
# 
# 1. Double DQN
# 2. Dueling DQN
# 3. Prioritized Experience Replay
# 4. Deep Recurrent Q-Network (DRQN)
# 5. Rainbow DQN
# 
# After training, plot the reward history curve and compare the performance of your improved DQN agent with the original DQN implementation.  
# 
# Next, test your improved DQN agent for 100 episodes and compute the average reward over these trials.  
# 
# You will receive 10 points only if both of the following conditions are met:
# 
# - The average reward is greater than 270
# - The average reward is higher than your original DQN agent
# 
# 
# This final evaluation will help you assess whether your modifications have effectively improved the agent's performance. 🚀

# ### DQN Variants

# In[7]:


# TODO: Implement your own DQN variant here, you may also need to create other classes
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=1):
        self.buffer = deque(maxlen=capacity)  # Double-Ended Queue
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        # ensure the newly added data will be sampled later
        max_priority = max(self.priorities) if self.priorities else 1.0
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
        self.priorities.append(max_priority)  # New experiences get max priority

    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority

    def sample(self, batch_size, beta=1):

      priorities = np.array(self.priorities, dtype=np.float32)
      scaled_priorities = priorities ** self.alpha
      probs = scaled_priorities / scaled_priorities.sum()

      batch_size = min(batch_size, len(self.buffer))
      indices = np.random.choice(len(self.buffer), batch_size, p=probs)
      experiences = [self.buffer[i] for i in indices]

      # Importance-sampling weights
      total = len(self.buffer)
      weights = (total * probs[indices]) ** (-beta)
      weights /= weights.max()  # Normalize

      weights = torch.tensor(weights, dtype=torch.float32)

      return {
        'states':      torch.tensor(np.array([e.state for e in experiences])),
        'actions':     torch.tensor(np.array([e.action for e in experiences])),
        'rewards':     torch.tensor(np.array([e.reward for e in experiences])),
        'next_states': torch.tensor(np.array([e.next_state for e in experiences])),
        'dones':       torch.tensor(np.array([e.done for e in experiences])),
        'weights': weights,
        'indices': indices  # for updating priorities later
      }

class DQNVariant(DQNAgent):
  def __init__(self, state_size, action_size):
      super().__init__(state_size, action_size)
      self.buffer = PrioritizedReplayBuffer(capacity=250000, alpha=1)

  def train(self, episode, episode_loss):

      sample = self.buffer.sample(batch_size=64)

      actions_taken = torch.tensor(sample['actions'], dtype=torch.long)
      action_distrs = self.train_net(sample['states'])
      preds = action_distrs[torch.arange(action_distrs.size(0)), actions_taken]

      # target from the target net
      max_next_Qs = self.target_net(sample['next_states']).max(dim=1).values
      max_next_Qs = max_next_Qs * (~sample['dones'])
      targets = sample['rewards'] + self.gamma * max_next_Qs
    #   loss = self.loss(preds, targets.to(torch.float))
      loss = (preds - targets.to(torch.float)).pow(2)
      loss = (sample['weights'] * loss).mean()

      # init priorities w the batch of TD error

      new_priorities = np.array([loss.detach().cpu().numpy()]) + 1e-5
      self.buffer.update_priorities(sample['indices'], new_priorities)

      episode_loss.append(loss.item())

      self.optim.zero_grad()
      loss.backward()
      # consider grad clipping here
      self.optim.step()

      # TODO: Update target network periodically
      # if step % 5 == 0:
      self.update(soft=True)



# ### Training the Improved DQN Agent

# In[ ]:


# TODO: Instantiate the agent and train it
agent = DQNVariant(state_size, action_size)

# TODO: Determine the number of episodes for training
num_episodes = 5000

reward_history = [] # Store the total rewards for each episode
eps_start = 1.0
eps_end = 0.1
eps_decay = 0.995
epsilon = eps_start
best_avg_rwd = 270
for episode in range(num_episodes):

    observation, info = env.reset()
    total_reward = 0
    done = False
    episode_loss = []
    while not done:

        action = agent.get_action(observation, epsilon)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.buffer.add(observation, action, reward, next_observation, done)

        agent.train(episode, episode_loss)


        total_reward += reward
        observation = next_observation

    # print(f"Episode {episode}, Reward: {total_reward}, avg_loss: {sum(episode_loss) / len(episode_loss)}")
    reward_history.append(total_reward)

    if (episode % 100 == 0) and (episode > 0):
      last_100_avg_rwd = sum(reward_history[-100:]) / len(reward_history[-100:])
      print(f"Episode {episode}, last 100 avg: {last_100_avg_rwd}")
      if last_100_avg_rwd > best_avg_rwd:
        print('saving weights')
        torch.save(agent.train_net.state_dict(), 'train_net.pth')
      if last_100_avg_rwd > 280:
        print("Early stopping")
        break
    epsilon = max(eps_end, epsilon*eps_decay)

# TODO: Plot the reward history to see how the agent's performance changes over time


# ### Discussion: Comparing the Performance of Two Agents (5%)  
# 
# Compare your improved DQN agent with the original in 150 words or less by addressing the following:
# 
# - What method(s) you used
# - Why it improved performance
# - Results comparison
# 
# TODO: Write down your answer here.

# ### Testing the Improved DQN Agent (10%)
# 
# You will receive full points only if both of the following conditions are met:
# 
# - The average reward is greater than 270
# - The average reward is higher than your original DQN agent
# 
# Do not modify the cell below, or you will get 0 points for this question.

# In[ ]:


# frames = [] # Uncomment to store the frames for the animation
reward_history = []
for _ in range(100):

    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state
        # frames.append(env.render()) # Uncomment to store the frames for the animation

    print(f"Reward: {total_reward}")
    reward_history.append(total_reward)

env.close()

print("Average reward:", np.mean(reward_history))

# Uncomment to show the animation
# gif_path = "DQN_variant.gif"
# imageio.mimsave(gif_path, frames, fps=30)
# Image(filename=gif_path)

