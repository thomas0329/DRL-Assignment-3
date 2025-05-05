import gym
from collections import deque
from actor import Mario
from env import make_env, GrayScaleObservation, ResizeObservation
import torch

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        
    def myinit(self):
        self.stack_sz = 4
        self.framestack = deque(maxlen=self.stack_sz)
        self.myenv = make_env()
        state_dim = self.myenv.observation_space.shape  # (4, 84, 84)
        action_dim = self.myenv.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = Mario(state_dim, action_dim, device=self.device)
        
    def act(self, observation):
        # train: state is the 3 prev frames and the current one
        # random actions during the first 4 frames of each episode
        # no action repeat
        if not hasattr(self, 'framestack'):
            self.myinit()
            
        observation = GrayScaleObservation.observation(observation)
        observation = ResizeObservation.observation(observation)
            
        if not self.framestack:
            # Fill the stack with k copies of the first frame
            for _ in range(4):
                self.framestack.append(observation)
        else:
            self.framestack.append(observation)
        
        # state: torch.Size([4, 84, 84])
        # no skip for the input stack
        self.agent.act()
        
        
        return self.action_space.sample()