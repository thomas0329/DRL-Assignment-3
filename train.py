
import torch
from collections import deque
from env import make_env
import numpy as np
from actor import Mario

def train(num_episodes=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    state_dim = env.observation_space.shape  # (4, 84, 84)
    action_dim = env.action_space.n          # usually 2 actions for Mario

    mario = Mario(state_dim, action_dim, device)

    rewards_history = []
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(np.array(state), dtype=torch.float)
        episode_reward = 0

        done = False
        while not done:
            print('eps', mario.epsilon)
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float)

            mario.cache((state, action, reward, next_state, done))
            mario.learn()

            state = next_state
            episode_reward += reward

        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-100:])
        print(f"Episode {episode+1} | Reward: {episode_reward:.1f} | Avg(100): {avg_reward:.1f} | Epsilon: {mario.epsilon:.2f}")
        

if __name__ == "__main__":
    train()

