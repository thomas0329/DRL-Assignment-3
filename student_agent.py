import gym
from collections import deque

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        
    def init_framestack(self):
        self.framestack = deque(maxlen=250000)
        
        
    def act(self, observation):
        # train: state is the 3 prev frames and the current one
        # random actions during the first 4 frames of each episode
        # no action repeat
        if not hasattr(self, 'framestack'):
            self.init_framestack()
            
        if not self.framestack:
            # Fill the stack with k copies of the first frame
            for _ in range(4):
                self.framestack.append(observation)
        else:
            self.framestack.append(observation)
        
        return self.action_space.sample()