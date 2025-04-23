from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from DQN import DQNAgent
import torch
from PIL import Image
import os
from utils import save_vid

# COMPLEX_MOVEMENT = [
#     ['NOOP'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
#     ['left', 'A'],
#     ['left', 'B'],
#     ['left', 'A', 'B'],
#     ['down'],
#     ['up'],
# ]

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
os.makedirs('frames', exist_ok=True)

# action space COMPLEX_MOVEMENT
# observation shape (240, 256, 3), numpy.ndarray
agent = DQNAgent(action_size=12)
num_episodes = 5

reward_history = [] # Store the total rewards for each episode
eps_start = 1.0
eps_end = 0.1
eps_decay = 0.995
epsilon = eps_start
for episode in range(num_episodes):

    observation = env.reset()
    total_reward = 0
    done = False
    episode_loss = []
    step = 0
    frames = []
    while not done:
        # TODO: Select an action from the agent
        action = agent.get_action(observation, epsilon)
        next_observation, reward, done, info = env.step(action)
        frame = env.render(mode='rgb_array')
        frames.append(frame)

        Image.fromarray(frame).save(f'frames/episode{episode}_step{step}.png')
        # TODO: Add the experience to the replay buffer and train the agent
        agent.buffer.add(observation, action, reward, next_observation, done)

        agent.train(episode, episode_loss, step)

        # TODO: Update the state and total reward
        total_reward += reward
        observation = next_observation
        step += 1

    save_vid(frames, episode)
    # print(f"Episode {episode}, Reward: {total_reward}, avg_loss: {sum(episode_loss) / len(episode_loss)}")
    reward_history.append(total_reward)

    if episode % 100 == 0:
      last_100_avg_rwd = sum(reward_history[-100:]) / len(reward_history[-100:])
      print(f"Episode {episode}, Avg of last 100 episodes: {last_100_avg_rwd}")
      if last_100_avg_rwd > 250:
        print('saving weights')
        torch.save(agent.train_net.state_dict(), 'train_net.pth')
      if last_100_avg_rwd > 260:
        print("Early stopping")
        break
    epsilon = max(eps_end, epsilon*eps_decay)


env.close()



