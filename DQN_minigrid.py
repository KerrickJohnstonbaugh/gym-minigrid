import argparse
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym_minigrid.simpleminigrid

seed = 4
env = gym.make('MiniGrid-Simple-DoorKey-8x8-v0')
env.seed(seed)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

from DQN.dqn_agent import Agent

agent = Agent(state_size=16, action_size=3, seed=0)


def dqn(args, n_episodes=1000, max_t=10000000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env.seed(seed)
        state = env.reset()
        score = 0
        step_count = 0
        while(1):
            step_count += 1
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tSteps: {:d}'.format(i_episode, np.mean(scores_window), step_count), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=0.75:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), '8x8checkpoint.pth')
            #break
        torch.save(agent.qnetwork_local.state_dict(), '8x8checkpoint.pth')
    return scores


def main(args):
    scores = dqn(args)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    agent.qnetwork_local.load_state_dict(torch.load('8x8checkpoint.pth'))

    for i in range(100):
        env.seed(seed)
        state = env.reset()
        ret = 0
        for j in range(1000):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            ret += reward
            if done:
                print(ret)
                break 
                
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="display reconstructed images")
    # copied args from train_autoencoder.py
    parser.add_argument('--exp_folder', type=str,
            help="directory where exp is, format: <path>/lightning_logs/version_<number>/checkpoints")
    args = parser.parse_args()
    main(args)