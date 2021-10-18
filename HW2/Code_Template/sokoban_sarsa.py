# -*- coding:utf-8 -*-
# Train Sarsa in Sokoban environment
import math, os, time, sys
import pdb
import numpy as np
import random, gym
from gym.wrappers import Monitor
from agent import SarsaAgent
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
num_space = 200000
agent = SarsaAgent(all_actions, num_actions, num_space)

# start training
for episode in range(1000):
    episode_reward = 0
    s = env.reset()
    state = s[0] + s[1] * 7 + ((s[2] + s[3] * 7)  + (s[4] + s[5] * 7) * 49 ) * 49
    a = agent.choose_action(state)
    # render env. You can comment all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # scalar s.
        state = s[0] + s[1] * 7 + ((s[2] + s[3] * 7)  + (s[4] + s[5] * 7) * 49 ) * 49
        s_, r, isdone, info = env.step(a)
        state_ = s_[0] + s_[1] * 7 + ((s_[2] + s_[3] * 7)  + (s_[4] + s_[5] * 7) * 49 ) * 49
        a_ = agent.choose_action(state_)
        # env.render()
        episode_reward += r
        # print(f"{s} {a} {s_} {r} {isdone}")
        agent.learn(state, state_, a, a_, r, gamma = 0.9)
        s = s_
        a = a_
        
        if isdone:
            # time.sleep(0.5)
            break
    if(agent.lr > 0.01):
        agent.lr *= 0.99
    # else:
        # agent.lr = 0

    if(agent.epsilon > 0.05):
        agent.epsilon *= 0.85
    else:
        agent.epsilon = 0
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  

    # At last, we should plot the episode reward during the train process.
    # We use excel to plot it because this may be seen nicely.
    # with open(r'HW2\\Code_Template\\episode_reward_sarsa_sokoban.txt', 'a', encoding='utf-8') as f:
    #     f.write(str(episode) + '\t' + str(episode_reward) + '\n')
    # Plot the epsilon value during the training.
    # with open(r'HW2\\Code_Template\\epsilon_value_sarsa_sokoban.txt', 'a', encoding='utf-8') as f:
        # f.write(str(episode) + '\t' + str(agent.epsilon) + '\n')
    # Visualize the final path after training.

print('\ntraining over\n')   

# close the render window after training.
env.close()

####### START CODING HERE #######





