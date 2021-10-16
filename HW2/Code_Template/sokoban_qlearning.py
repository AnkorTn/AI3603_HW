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
# add paras size of action space
agent = QLearningAgent(all_actions, num_actions)

# start training
for episode in range(1000):
    episode_reward = 0
    s = env.reset()
    # render env. You can comment all render() to turn off the GUI to accelerate training.
    env.render()
    # agent interacts with the environment
    for iter in range(500):
        # convert coordinate to index
        idx_s0 = s[0] + 7*s[1]
        idx_s1 = s[2] + 7*s[3]
        idx_s2 = s[4] + 7*s[5]

        # a = agent.choose_action(idx_s1)
        a = agent.choose_action(idx_s0)
        s_, r, isdone, info = env.step(a)

        # convert coordinate to index
        idx_s_0 = s_[0] + 7*s_[1]
        idx_s_1 = s_[2] + 7*s_[3]
        idx_s_2 = s_[4] + 7*s_[5]

        env.render()
        episode_reward += r
        # print(f"{s} {a} {s_} {r} {isdone}")
        # agent.learn(idx_s1, idx_s_1, a, r, gamma=0.9)
        agent.learn(idx_s0, idx_s_0, a, r, gamma=0.9)
        s = s_
        if isdone:
            # # assign the value of teminal
            # for a in all_actions:
            #     agent.q_table[idx_s1][a] = r

            # time.sleep(0.5)
            break

    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  

    # decrease epsilon and learning rate
    if (agent.epsilon < 1e-2):
        agent.epsilon = 0
    else:
        agent.epsilon *= 0.99

    agent.lr *= 0.99

print('\ntraining over\n')   

# close the render window after training.
env.close()

####### START CODING HERE #######





