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
num_space = 200000
agent = QLearningAgent(all_actions, num_actions, num_space)

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
        # if(r > 0):
            # print(f"{s} {a} {s_} {r} {isdone}")
        agent.learn(state, state_, a, r, gamma=0.9)
        # print("Q_TABLE")
        # print(agent.q_table)
        # time.sleep(0.5)
        s = s_
        if isdone:
            # for a in all_actions:
                # agent.q_table[_s][a] = 0
            break

    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  

    # decrease epsilon and learning rate
    if (agent.epsilon < 0.05):
        agent.epsilon = 0
    else:
        # exponential decay 
        agent.epsilon *= 0.9
        # linear decay
        # agent.epsilon -= 0.01
    
    # segmented attenuation
    # if (episode < 20):
    #     agent.epsilon = 1
    # elif (episode < 40):
    #     agent.epsilon = 0.9
    # elif (episode < 60):
    #     agent.epsilon = 0.8
    # elif (episode < 80):
    #     agent.epsilon = 0.7
    # elif (episode < 100):
    #     agent.epsilon = 0.6
    # elif (episode < 120):
    #     agent.epsilon = 0.5
    # elif (episode < 140):
    #     agent.epsilon = 0.4
    # elif (episode < 160):
    #     agent.epsilon = 0.3
    # elif (episode < 180):
    #     agent.epsilon = 0.2
    # elif (episode < 200):
    #     agent.epsilon = 0.1
    # else:
    #     agent.epsilon = 0

    agent.lr *= 0.95

    # At last, we should plot the episode reward during the train process.
    # We use excel to plot it because this may be seen nicely.
    with open(r'd:\Users\86134\Documents\github\AI3603_HW\AI3603_HW\HW2\Code_Template\episode_reward_qlearning_sokoban.txt', 'a', encoding='utf-8') as f:
        f.write(str(episode) + '\t' + str(episode_reward) + '\n')
    # Plot the epsilon value during the training.
    with open(r'd:\Users\86134\Documents\github\AI3603_HW\AI3603_HW\HW2\Code_Template\epsilon_value_qlearning_sokoban.txt', 'a', encoding='utf-8') as f:
        f.write(str(episode) + '\t' + str(agent.epsilon) + '\n')
    # Visualize the final path after training.

print('\ntraining over\n')   

# close the render window after training.
env.close()

####### START CODING HERE #######





