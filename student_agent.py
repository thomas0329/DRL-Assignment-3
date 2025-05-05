import gym
from collections import deque
from actor import Mario
from env import make_env, GrayScaleObservation, ResizeObservation
import torch
import numpy as np

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        
    # env = SkipFrame(env, skip=4)
    # env = GrayScaleObservation(env)
    # env = ResizeObservation(env, shape=84)
    # env = FrameStack(env, num_stack=4)
    
    def myinit(self):
        self.act_counter = 0
        self.action = None
        
        self.stack_sz = 4
        self.framestack = deque(maxlen=self.stack_sz)
        self.myenv = make_env()
        self.gray = GrayScaleObservation(self.myenv)
        self.resize = ResizeObservation(self.myenv, shape=84)
        state_dim = self.myenv.observation_space.shape  # (4, 84, 84)
        action_dim = self.myenv.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = Mario(state_dim, action_dim, device=self.device)
        self.agent.train_net.load_state_dict(torch.load('train_net.pth'))
        self.agent.train_net.eval()
        
    def add_frame(self, observation):
        if not self.framestack:
            # Fill the stack with k copies of the first frame
            for _ in range(4):
                self.framestack.append(observation)
        else:
            self.framestack.append(observation)
        
    def act(self, observation):
        # train: state is the 3 prev frames and the current one
        # random actions during the first 4 frames of each episode
        if not hasattr(self, 'agent'):
            self.myinit()
            
        if self.act_counter % 4 == 0:   
            # decide a new action 
            observation = self.gray.observation(observation)
            observation = self.resize.observation(observation)
            self.add_frame(observation)
            state = np.stack(self.framestack, axis=0)  # shape: (4, 84, 84)
            state = torch.tensor(np.array(state), dtype=torch.float32) / 255.0
            action = self.agent.act(state)
            self.action = action
            
        else:
            # repeat action
            action = self.action
        
        self.act_counter += 1
        return action