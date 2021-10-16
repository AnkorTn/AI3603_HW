# -*- coding:utf-8 -*-
# Train Sarsa in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from gym_gridworld import CliffWalk
from agent import SarsaAgent 
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

####### START CODING HERE #######

# construct the intelligent agent.
agent = SarsaAgent(all_actions, num_actions)

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # choose an action
    a = agent.choose_action(s)
    # render env. You can comment all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # take action a, observe r, s_
        s_, r, isdone, info = env.step(a)
        # choose a_ from s_ using policy derived from Q
        a_ = agent.choose_action(s_)
        # env.render()
        # update the episode reward
        episode_reward += r
        # print(f"{s} {a} {s_} {r} {isdone}")
        # if(episode == 950):
            # print(f"{s} {a} {s_} {r} {isdone}")
        # agent learns from experience
        agent.learn(s, s_, a, a_, r, gamma = 0.9)
        s = s_
        a = a_
        if isdone:
            # assign the value of teminal
            for a in all_actions:
                agent.q_table[s][a] = r
            
            # time.sleep(0.5)
            break    
    # the epsilon value declines in each step.
    if(agent.lr > 0.01):
        agent.lr *= 0.95
    # else:
        # agent.lr = 0

    if(agent.epsilon > 0.008):
        agent.epsilon *= 0.95
    else:
        agent.epsilon = 0
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  
    # At last, we should plot the episode reward during the train process.
    # We use excel to plot it because this may be seen nicely.
    with open(r'HW2\\Code_Template\\episode_reward_sarsa.txt', 'a', encoding='utf-8') as f:
        f.write(str(episode) + '\t' + str(episode_reward) + '\n')
    # Plot the epsilon value during the training.
    with open(r'HW2\\Code_Template\\epsilon_value_sarsa.txt', 'a', encoding='utf-8') as f:
        f.write(str(episode) + '\t' + str(agent.epsilon) + '\n')
    # Visualize the final path after training.
print('\ntraining over\n')

# close the render window after training.
env.close()

####### START CODING HERE #######


