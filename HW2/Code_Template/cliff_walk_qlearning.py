# -*- coding:utf-8 -*-
# Train Q-Learning in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from gym_gridworld import CliffWalk
from agent import QLearningAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# construct the environment
env = CliffWalk()
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

##### START CODING HERE #####

# construct the intelligent agent.
# num_space
num_space = 4 * 12
# add paras size of action space
agent = QLearningAgent(all_actions, num_actions, num_space)

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # render env. You can comment all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # choose an action
        a = agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        # env.render()
        # update the episode reward
        episode_reward += r
        
        # if(episode == 950):
            # print(f"{s} {a} {s_} {r} {isdone}")
        # agent learns from experience

        # add paras size of action space, learning rate and epsilon (for epsilon-greedy)
        agent.learn(s, s_, a, r, gamma=0.9)
        s = s_
        if isdone:
            # assign the value of teminal
            for a in all_actions:
                agent.q_table[s][a] = r

            # time.sleep(0.5)
            break
    # print(agent.q_table)

    print('episode:{} episode_reward:{} epsilon:{}'.format(episode,episode_reward,round(agent.epsilon,1)))
    # print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  

    # decrease epsilon and learning rate
    if (agent.epsilon < 1e-2):
        agent.epsilon = 0
    else:
        agent.epsilon *= 0.95
    agent.lr *= 0.99


    # At last, we should plot the episode reward during the train process.
    # We use excel to plot it because this may be seen nicely.
    # with open(r'HW2\\Code_Template\\episode_reward_Q_learning.txt', 'a', encoding='utf-8') as f:
        # f.write(str(episode) + '\t' + str(episode_reward) + '\n')
    # Plot the epsilon value during the training.
    # with open(r'HW2\\Code_Template\\epsilon_value_Q_learning.txt', 'a', encoding='utf-8') as f:
        # f.write(str(episode) + '\t' + str(agent.epsilon) + '\n')

# print('\ntraining over\n')   

# close the render window after training.
env.close()

##### START CODING HERE #####


