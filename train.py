from ICM import ICM
import torch
from collections import deque
from env import make_env
import numpy as np
from actor import Mario
# import imageio
import os
from utils import save_gif
from PIL import Image
import logging
from datetime import datetime
import wandb

# Setup logging
# Create a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
log_filename = f"logs/{timestamp}/train_log.log"
dir = f"logs/{timestamp}"
os.makedirs(dir, exist_ok=True)

logging.basicConfig(
    filename=log_filename,
    filemode="w",  # overwrite every time; use 'a' to append
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def train(config, logger, gif_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    state_dim = env.observation_space.shape  # (4, 84, 84)
    action_dim = env.action_space.n          

    agent = Mario(state_dim, action_dim, device)
    # icm = ICM()

    rewards_history = []
    best_avg_reward = 0
    for episode in range(config["num_episodes"]):
        state = env.reset()
        state = torch.tensor(np.array(state), dtype=torch.float32) / 255.0
        # state = torch.tensor(np.array(state), dtype=torch.float)
        # print('state', state.shape)
        episode_reward, episode_reward_ex = 0, 0

        done = False
        frames = []
        step = 0
        while True:
            
            next_state, reward, done = agent.n_step_act(env, state, frames, epsgreedy=config["epsgreedy"])
            # action = agent.act(state, epsgreedy=config["epsgreedy"])
            # next_state, reward_e, done, _ = env.step(action)

            agent.update()
            episode_reward += reward
            step += 1
            
            if done or step > config["max_steps"]:
                break

            state = next_state
            
        save_gif(frames, episode, gif_dir)
        rewards_history.append(episode_reward)
        
        avg_reward = np.mean(rewards_history[-100:])

        print(f"Episode {episode+1} | Reward total: {episode_reward:.1f} | Avg(100): {avg_reward:.1f} | buffer len: {len(agent.replay_buffer)}/{agent.replay_buffer.maxlen} | epsilon: {agent.epsilon}")

        logger.log({
            "episode": episode,
            # "reward_ex": episode_reward_ex,
            # "reward_total": episode_reward,
            "avg_reward": avg_reward,
            # "avg_reward_ex": avg_reward_ex,
            "buffer_len": len(agent.replay_buffer),

            "noise1": agent.train_net.noisy1.weight_sigma.mean(),
            "noise2": agent.train_net.noisy2.weight_sigma.mean(),
        }, step=episode)
        if avg_reward > best_avg_reward:
            torch.save(agent.train_net.state_dict(), "train_net.pth")

        

if __name__ == "__main__":
    
    wandb.init(project="mario", config={
        "num_episodes": 5000, 
        "max_steps": 2500,
        "epsgreedy": False
    })
    config = wandb.config
    
    # logger = logging.getLogger()
    gif_dir = os.path.join(wandb.run.dir, "gameplay_gifs")
    train(config, wandb, gif_dir)

