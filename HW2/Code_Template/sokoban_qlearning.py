# -*- coding:utf-8 -*-
# Train Q-Learning in Sokoban environment
import math, os, time, sys
import pdb
import numpy as np
import random, gym
from gym.wrappers import Monitor
from agent import QLearningAgent
import gym_sokoban
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####


# construct the environment
env = gym.make('Sokoban-hw2-v0')
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 


####### START CODING HERE #######

# construct the intelligent agent.
agent = QLearningAgent(all_actions, num_actions)

# start training
for episode in range(1000):
    episode_reward = 0
    s = env.reset()
    # render env. You can comment all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # scalar s.
        _s = s[0] + s[1] * 7
        a = agent.choose_action(_s)
        s_, r, isdone, info = env.step(a)
        # scalar s__.
        s__ = s_[0] + s_[1] * 7

        # env.render()
        episode_reward += r
        # if(r > 0):
            # print(f"{s} {a} {s_} {r} {isdone}")
        agent.learn(_s, s__, a, r, gamma=0.9)
        # print("Q_TABLE")
        # print(agent.q_table)
        # time.sleep(0.5)
        s = s_
        if(r > 0.8):
            print(f"{s} {a} {s_} {r} {isdone}")
            print("This is the reward.\n")
            for j in all_actions:
                agent.q_table[_s][j] = 100
            if(r > 8):
                for j in all_actions:
                    agent.q_table[_s][j] = 100
        if isdone:
            # for a in all_actions:
                # agent.q_table[_s][a] = 0
            break

    if (agent.epsilon < 1e-2):
        agent.epsilon = 0
    else:
        agent.epsilon *= 0.99
    # agent.lr *= 0.99

    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  
print('\ntraining over\n')   

# close the render window after training.
env.close()

####### START CODING HERE #######





