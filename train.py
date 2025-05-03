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

    rewards_history, rewards_history_ex = [], []
    for episode in range(config["num_episodes"]):
        state = env.reset()
        state = torch.tensor(np.array(state), dtype=torch.float)
        episode_reward, episode_reward_ex = 0, 0

        done = False
        frames = []
        
        while True:
            # state: 1, 2, 3, curr=4
            action = agent.act(state, epsgreedy=config["epsgreedy"])
            # import pdb; pdb.set_trace()
            # 4 action repeat
            next_state, reward_e, done, _ = env.step(action)
            # next_state: 5, 6, 7, curr=8
            frame = env.render(mode='rgb_array')  # returns RGB frame (H, W, 3)
            # frame (240, 256, 3)
            frames.append(Image.fromarray(frame))
            
            # reward_i = icm(state, next_state, action)
            # print('reward e', reward_e, 'reward i', reward_i)
            reward = reward_e
            # reward = reward_e + reward_i
            
            next_state = torch.tensor(np.array(next_state), dtype=torch.float)

            agent.save((state, action, reward, next_state, done), logger)
            agent.update()

            state = next_state
            episode_reward_ex += reward_e
            episode_reward += reward
            
            if done:
                break
        
        save_gif(frames, episode, gif_dir)
        rewards_history.append(episode_reward)
        rewards_history_ex.append(episode_reward_ex)
        avg_reward = np.mean(rewards_history[-100:])
        avg_reward_ex = np.mean(rewards_history_ex[-100:])
        # logger.log(f"Episode {episode+1} | Reward ex: {episode_reward_ex:.1f} | Reward total: {episode_reward:.1f} | Avg(100): {avg_reward:.1f} | Avg Ex(100): {avg_reward_ex:.1f} | buffer len: {len(agent.replay_buffer)}/{agent.replay_buffer.maxlen}")
        logger.log({
            "episode": episode,
            # "reward_ex": episode_reward_ex,
            # "reward_total": episode_reward,
            "avg_reward": avg_reward,
            # "avg_reward_ex": avg_reward_ex,
            # "buffer_len": len(agent.replay_buffer)
        }, step=episode)
        

        

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

