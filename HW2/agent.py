# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
import random
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

"""
Instruction: 
Currently, the following agents are `random` policy.
You need to implement the Q-learning agent, Sarsa agent and Dyna-Q agent in this file.
"""

# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Sarsa agent here"""
class SarsaAgent(object):
    ##### START CODING HERE #####
    # add paras size of action space, and learning rate
    def __init__(self, all_actions, num_actions, num_space):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions

        # epsilon-greedy
        self.epsilon = 1.0

        # initialize Q-table by the size of action space and state space (4*12)
        self.q_table = [[0 for _ in range(num_actions)] for _ in range(num_space)]

        # learning rate
        self.lr = 1.0

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        # epsilon-greedy
        # epsilon probability to choose random strategy
        if (random.random() < self.epsilon):
            action = np.random.choice(self.all_actions)
            return action

        # (1-epsilon) probability to choose determine strategy
        # calculate pi according to Q and exploration strategy
        action = 0

        max_q = float('-inf')

        for i in range(len(self.q_table[observation])):
            if (self.q_table[observation][i] > max_q):
                max_q = self.q_table[observation][i]
                action = i

        return action
    
    # add paras previous state, next state, action, reward and discounting factor
    # Sarsa: firstly determine the direction and then go directly.
    def learn(self, s, s_, a, a_, r, gamma):
        """learn from experience"""
        # time.sleep(0.5)
        # q_next()
        max_q = self.q_table[s_][a_]

        # Sarsa-algorithm update rule
        # topo in pseudocode
        self.q_table[s][a] = (1-self.lr)*self.q_table[s][a] + self.lr*(r + gamma*max_q)

        # print("[INFO] The learning process complete. (???????????)???")
        return True
    ##### END CODING HERE #####
# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Q-Learning agent here"""
class QLearningAgent(object):
    ##### START CODING HERE #####
    # add paras size of action space
    def __init__(self, all_actions, num_actions, num_space):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions

        # epsilon-greedy
        self.epsilon = 1.0

        # initialize Q-table by the size of action space and state space (4*12)
        self.q_table = [[0 for _ in range(num_actions)] for _ in range(num_space)]

        # learning rate
        self.lr = 1.0

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        # epsilon-greedy
        # epsilon probability to choose random strategy
        if (random.random() < self.epsilon):
            action = np.random.choice(self.all_actions)
            return action

        # (1-epsilon) probability to choose determine strategy
        # calculate pi according to Q and exploration strategy
        action = 0

        max_q = float('-inf')

        for i in range(len(self.q_table[observation])):
            if (self.q_table[observation][i] > max_q):
                max_q = self.q_table[observation][i]
                action = i

        return action
    
    # add paras previous state, next state, action, reward and discounting factor
    def learn(self, s, s_, a, r, gamma):
        """learn from experience"""
        # time.sleep(0.5)

        max_q = float('-inf')

        for i in range(len(self.q_table[s_])):
            if (self.q_table[s_][i] > max_q):
                max_q = self.q_table[s_][i]

        # Q-learning update rule
        # topo in pseudocode
        self.q_table[s][a] = (1-self.lr)*self.q_table[s][a] + self.lr*(r + gamma*max_q)

        # print("[INFO] The learning process complete. (???????????)???")
        return True
    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Dyna-Q agent here"""
class DynaQAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, num_actions, num_space):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions

        # epsilon-greedy
        self.epsilon = 1.0

        # initialize Q-table by the size of action space and state space (4*12)
        self.q_table = [[0 for _ in range(num_actions)] for _ in range(num_space)]

        # learning rate
        self.lr = 1.0

        # each experience contains a tuple (s,a,s',r)
        self.experience = []

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        # epsilon-greedy
        # epsilon probability to choose random strategy
        if (random.random() < self.epsilon):
            action = np.random.choice(self.all_actions)
            return action

        # (1-epsilon) probability to choose determine strategy
        # calculate pi according to Q and exploration strategy
        action = 0

        max_q = float('-inf')

        for i in range(len(self.q_table[observation])):
            if (self.q_table[observation][i] > max_q):
                max_q = self.q_table[observation][i]
                action = i

        return action
    
    # add paras previous state, next state, action, reward and discounting factor
    def learn(self, s, s_, a, r, gamma):
        """learn from experience"""
        # time.sleep(0.5)

        max_q = float('-inf')

        for i in range(len(self.q_table[s_])):
            if (self.q_table[s_][i] > max_q):
                max_q = self.q_table[s_][i]

        # Q-learning update rule
        # topo in pseudocode
        self.q_table[s][a] = (1-self.lr)*self.q_table[s][a] + self.lr*(r + gamma*max_q)

        # print("[INFO] The learning process complete. (???????????)???")
        return True
    
    # prioritized sample
    def TD_error(self, s, s_, a, r, gamma):
        """You can add other functions as you wish."""
        max_q = float('-inf')

        for i in range(len(self.q_table[s_])):
            if (self.q_table[s_][i] > max_q):
                max_q = self.q_table[s_][i]

        td = r + gamma*max_q - self.q_table[s][a]
        return td

    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: (optional) Implement RL agent(s) with other exploration methods you have found"""
##### START CODING HERE #####
class RLAgentWithOtherExploration(object):
    """initialize the agent"""
    def __init__(self, all_actions, num_actions, num_space):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions

        # initialize Q-table by the size of action space and state space (4*12)
        self.q_table = [[0 for _ in range(num_actions)] for _ in range(num_space)]

        # record the number of selection times.
        # num_action has 0,1,2,3 up to 4 directions, and the "4" dimension is the total number.
        self.times_table = [[0 for _ in range(num_actions + 1)] for _ in range(num_space)]

        # learning rate
        self.lr = 1.0

    def choose_action(self, observation):
        """choose action with other exploration algorithms."""
        # calculate pi according to Q and exploration strategy
        action = 0

        max_q = float('-inf')

        for i in range(len(self.q_table[observation])):
            if(self.times_table[observation][i] == 0):
                action = i
                break
            else:
                temp = self.q_table[observation][i] + math.sqrt(2 * math.log(self.times_table[observation][4])/self.times_table[observation][i])
                if (temp > max_q):
                    max_q = temp
                    action = i
        self.times_table[observation][4] += 1
        self.times_table[observation][action] += 1
        return action
    
    def learn(self, s, s_, a, r, gamma):
        """learn from experience"""
        # time.sleep(0.5)

        max_q = float('-inf')

        for i in range(len(self.q_table[s_])):
            if (self.q_table[s_][i] > max_q):
                max_q = self.q_table[s_][i]

        # Q-learning update rule
        # topo in pseudocode
        self.q_table[s][a] = (1-self.lr)*self.q_table[s][a] + self.lr*(r + gamma*max_q)

        # print("[INFO] The learning process complete. (???????????)???")
        return True
##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #