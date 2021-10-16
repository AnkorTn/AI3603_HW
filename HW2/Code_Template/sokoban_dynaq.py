# -*- coding:utf-8 -*-
# Train Dyna-Q in Sokoban environment
import math, os, time, sys
import pdb
import numpy as np
import random, gym
from gym.wrappers import Monitor
from agent import DynaQAgent
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
agent = DynaQAgent(all_actions, num_actions, num_space=200000)

# start training
for episode in range(1000):
    episode_reward = 0
    s = env.reset()
    # render env. You can comment all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # scalar s.
        state = s[0] + s[1] * 7 + ((s[2] + s[3] * 7)  + (s[4] + s[5] * 7) * 49 ) * 49

        a = agent.choose_action(state)
        s_, r, isdone, info = env.step(a)
        # scalar s__.
        state_ = s_[0] + s_[1] * 7 + ((s_[2] + s_[3] * 7)  + (s_[4] + s_[5] * 7) * 49 ) * 49

        # env.render()
        episode_reward += r

        # append experience into experience pool
        agent.experience.add((state, state_, a, r))

        print(len(agent.experience))

        # print(f"{s} {a} {s_} {r} {isdone}")

        # learn from experience pool
        for experience in agent.experience:
            state, state_ = experience[0], experience[1]
            a = experience[2]
            r = experience[3]
            agent.learn(state, state_, a, r, gamma=0.9)
        
        s = s_
        if isdone:
            # time.sleep(0.5)
            break

    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  

    # decrease epsilon and learning rate
    if (agent.epsilon < 0.05):
        agent.epsilon = 0
    else:
        agent.epsilon *= 0.97

    agent.lr *= 0.99


print('\ntraining over\n')   

# close the render window after training.
env.close()

####### START CODING HERE #######





