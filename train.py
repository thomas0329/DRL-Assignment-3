from ICM import ICM
import torch
from collections import deque
from env import make_env
import numpy as np
from actor import Mario
import imageio
import os
from utils import save_gif
from PIL import Image

def train(num_episodes=500, max_steps=2500, epsgreedy=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    state_dim = env.observation_space.shape  # (4, 84, 84)
    action_dim = env.action_space.n          

    agent = Mario(state_dim, action_dim, device)
    # icm = ICM()

    rewards_history = []
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(np.array(state), dtype=torch.float)
        episode_reward, episode_reward_ex = 0, 0

        done = False
        frames = []
        # for _ in range(max_steps):
        while True:
            action = agent.act(state, epsgreedy=epsgreedy)
            next_state, reward_e, done, _ = env.step(action)
            frame = env.render(mode='rgb_array')  # returns RGB frame (H, W, 3)
            frames.append(Image.fromarray(frame))
            
            # reward_i = icm(state, next_state, action)
            # print('reward e', reward_e, 'reward i', reward_i)
            reward = reward_e
            # reward = reward_e + reward_i
            
            next_state = torch.tensor(np.array(next_state), dtype=torch.float)

            agent.save((state, action, reward, next_state, done))
            agent.update()

            state = next_state
            episode_reward_ex += reward_e
            episode_reward += reward
            
            if done:
                break
        
        save_gif(frames, episode)
        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-100:])
        print(f"Episode {episode+1} | Reward ex: {episode_reward_ex:.1f} | Reward total: {episode_reward:.1f} | Avg(100): {avg_reward:.1f}")
        

if __name__ == "__main__":
    train()

