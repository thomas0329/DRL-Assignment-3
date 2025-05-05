import gym
from collections import deque
from actor import Mario
from env import make_env, GrayScaleObservation, ResizeObservation
import torch
import numpy as np
import cv2  # OpenCV for grayscale + resize

# # Do not modify the input of the 'act' function and the '__init__' function. 
# class Agent(object):
#     """Agent that acts randomly."""
#     def __init__(self):
#         self.action_space = gym.spaces.Discrete(12)
    
#     # env = SkipFrame(env, skip=4)
#     # env = GrayScaleObservation(env)
#     # env = ResizeObservation(env, shape=84)
#     # env = FrameStack(env, num_stack=4)
    
#     def myinit(self):
#         self.act_counter = 0
#         self.action = None
        
#         self.stack_sz = 4
#         self.framestack = deque(maxlen=self.stack_sz)
#         self.myenv = make_env()
#         self.gray = GrayScaleObservation(self.myenv)
#         self.resize = ResizeObservation(self.myenv, shape=84)
#         state_dim = self.myenv.observation_space.shape  # (4, 84, 84)
#         action_dim = self.myenv.action_space.n
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.agent = Mario(state_dim, action_dim, device=self.device)
#         self.agent.train_net.load_state_dict(torch.load('train_net.pth'))
#         self.agent.train_net.eval() # disable noise
        
#     def add_frame(self, observation):
#         if not self.framestack:
#             # Fill the stack with k copies of the first frame
#             for _ in range(4):
#                 self.framestack.append(observation)
#         else:
#             self.framestack.append(observation)
        
#     def act(self, observation):
#         print('observation', observation.shape)
#         # train: state is the 3 prev frames and the current one
#         # random actions during the first 4 frames of each episode
#         if not hasattr(self, 'agent'):
#             self.myinit()
            
#         if self.act_counter % 4 == 0:   
#             # decide a new action 
#             observation = self.gray.observation(observation)
#             observation = self.resize.observation(observation)
#             self.add_frame(observation)
#             state = np.stack(self.framestack, axis=0)  # shape: (4, 84, 84)
#             state = torch.tensor(np.array(state), dtype=torch.float32) / 255.0
#             action = self.agent.act(state)
#             self.action = action
            
#         else:
#             # repeat action
#             action = self.action
        
#         self.act_counter += 1
#         return action
    
class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

    def myinit(self):
        self.act_counter = 0
        self.action = None
        self.stack_sz = 4
        self.framestack = deque(maxlen=self.stack_sz)

        # Create dummy env to get shape info
        dummy_obs = self.preprocess(np.zeros((240, 256, 3), dtype=np.uint8))
        state_dim = (self.stack_sz, *dummy_obs.shape)
        action_dim = self.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = Mario(state_dim, action_dim, device=self.device)
        self.agent.train_net.load_state_dict(torch.load('train_net.pth', map_location=self.device))
        self.agent.train_net.eval()  # disable noise, dropout, etc.

    def preprocess(self, obs):
        # Convert to grayscale and resize to (84, 84)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # Shape: (240, 256) → (H, W)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized  # Shape: (84, 84)

    def add_frame(self, frame):
        if not self.framestack:
            for _ in range(self.stack_sz):
                self.framestack.append(frame)
        else:
            self.framestack.append(frame)

    def act(self, observation):
        if not hasattr(self, 'agent'):
            self.myinit()

        if self.act_counter % 4 == 0:
            # Preprocess and update frame stack
            frame = self.preprocess(observation)
            self.add_frame(frame)

            # Stack to shape (4, 84, 84)
            state = np.stack(self.framestack, axis=0).astype(np.float32) / 255.0
            state = torch.tensor(state).to(self.device)  # (1, 4, 84, 84)

            with torch.no_grad():
                action = self.agent.act(state)

            self.action = action  # Save for next 3 steps
        else:
            action = self.action  # Repeat action

        self.act_counter += 1
        return action
