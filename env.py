
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from torchvision import transforms as T
import torch
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# class Nstep(gym.Wrapper):
#     def __init__(self, env, step):
#         super().__init__(env)
#         self.c = step
#         self.n_step_buffer = np.zeros((2,) + self.env.observation_space.shape)

#     def step(self, action):
        
#         total_reward = 0.0
#         for i in range(self.n_step):
#             obs, reward, done, info = self.env.step(action)
            
#             if i == self.skip - 2:
#                 self.obs_buffer[0] = obs
#             if i == self.skip - 1:
#                 self.obs_buffer[1] = obs
            
#             total_reward += reward
#             if done:
#                 break

#         max_obs = np.maximum(self.obs_buffer[0], self.obs_buffer[1])
#         return max_obs, total_reward, done, info

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
        self.obs_buffer = np.zeros((2,) + self.env.observation_space.shape)
        
    def step(self, action):
        
        total_reward = 0.0
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            
            if i == self.skip - 2:
                self.obs_buffer[0] = obs
            if i == self.skip - 1:
                self.obs_buffer[1] = obs
            
            total_reward += reward
            if done:
                break

        max_obs = np.maximum(self.obs_buffer[0], self.obs_buffer[1])
        return max_obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        obs = np.transpose(observation, (2, 0, 1))
        obs = torch.tensor(obs.copy(), dtype=torch.float)
        return T.Grayscale()(obs)

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        t = T.Compose([T.Resize(self.shape, antialias=True), T.Normalize(0, 255)])
        return t(observation).squeeze(0)
        # return t(torch.from_numpy(observation)).squeeze(0)
        

def make_env():
    
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    
    return env


