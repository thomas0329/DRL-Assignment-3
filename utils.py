import numpy as np
import torch
# import imageio
import gym
from gym.spaces import Box
from torchvision import transforms as T
import numpy as np
import os
import random
import numpy as np

import numpy as np
import random

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.size = 0

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get_leaf(self, v):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if v <= self.tree[left]:
                idx = left
            else:
                v -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5  # for non-zero priority
        self.max_priority = 1.0
        self.maxlen = capacity

    def add(self, transition):
        self.tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(v)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        probs = np.array(priorities) / self.tree.total_priority()
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.uint8),
            idxs,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, idxs, priorities):
        for idx, p in zip(idxs, priorities):
            self.tree.update(idx, (p + self.epsilon) ** self.alpha)
            self.max_priority = max(self.max_priority, p + self.epsilon)
            
    def __len__(self):
        return self.tree.size



def save_gif(frames, episode, save_dir='./'):
    os.makedirs(f'{save_dir}/gifs', exist_ok=True)
    gif_path = f'{save_dir}/gifs/episode_{episode}.gif'

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=40,  # duration per frame in ms (~25 FPS)
        loop=0
    )

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation




def add_frame(input_stack, frame):
    input_stack.append(frame)
    return make_frame_stack(list(input_stack))


def make_frame_stack(frames, required_stack=4):
    """
    Ensures a stack of exactly `required_stack` frames.
    If fewer than required, repeat the first frame.

    Args:
        frames (List[np.ndarray]): List of frames, each of shape (H, W, 3)
        required_stack (int): Number of frames to stack (default: 4)

    Returns:
        np.ndarray: Array of shape (required_stack, H, W, 3)
    """
    assert len(frames) > 0, "At least one frame is required."
    stack = frames.copy()
    while len(stack) < required_stack:
        stack.insert(0, stack[0])  # Repeat the first frame
    stack = stack[-required_stack:]  # Keep only the latest N frames
    return np.stack(stack, axis=0)

