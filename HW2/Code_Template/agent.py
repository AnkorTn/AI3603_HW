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
    def __init__(self, all_actions):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 1.0


    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        action = np.random.choice(self.all_actions)
        return action
    
    def learn(self):
        """learn from experience"""
        time.sleep(0.5)
        print("[INFO] The learning process complete. (ﾉ｀⊿´)ﾉ")
        return True
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        do_something = True
        return None

    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Q-Learning agent here"""
class QLearningAgent(object):
    ##### START CODING HERE #####
    # add paras size of action space, learning rate and epsilon (for epsilon-greedy)
    def __init__(self, all_actions, num_actions, lr, epsilon):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 1.0

        # initialize Q-table by the size of action space and state space (4*12)
        self.q_table = [[0 for _ in range(num_actions)] for _ in range(4*12)]

        # assign the value of goal
        for j in range(4):
            self.q_table[11][j] = 10

        # assign the value of cliff
        for i in range(1,11):
            for j in range(4):
                self.q_table[i][j] = -100

        # learning rate
        self.lr = lr

        # epsilon-greedy
        self.epsilon = epsilon

    def your_function(self,s):
        """determine whether s is in the terminal state """
        """You can add other functions as you wish."""
        # reach the goal or fall into the cliff
        return (s >= 1 and s <= 11)

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        # epsilon-greedy
        # epsilon probability to choose random strategy
        if (random.random() < self.epsilon):
            action = np.random.choice(self.all_actions)
            return action

        # (1-epsilon) probability to choose determine strategy 
        # if s is not terminal
        if (not self.your_function(observation)):
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

        # if s is terminal
        if (self.your_function(s_)):
            self.q_table[s][a] = self.q_table[s_][0]
            return

        max_q = float('-inf')

        for i in range(len(self.q_table[s_])):
            if (self.q_table[s_][i] > max_q):
                max_q = self.q_table[s_][i]

        # Q-learning update rule
        # topo in pseudocode
        self.q_table[s][a] = (1-self.lr)*self.q_table[s][a] + self.lr*(r + gamma*max_q)

        # print("[INFO] The learning process complete. (ﾉ｀⊿´)ﾉ")
        return True
    
    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Dyna-Q agent here"""
class DynaQAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 1.0

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        action = np.random.choice(self.all_actions)
        return action
    
    def learn(self):
        """learn from experience"""
        time.sleep(0.5)
        print("[INFO] The learning process complete. (ﾉ｀⊿´)ﾉ")
        return True
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        do_something = True
        return None

    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: (optional) Implement RL agent(s) with other exploration methods you have found"""
##### START CODING HERE #####
class RLAgentWithOtherExploration(object):
    """initialize the agent"""
    def __init__(self, all_actions):
        self.all_actions = all_actions
        self.epsilon = 1.0

    def choose_action(self, observation):
        """choose action with other exploration algorithms."""
        action = np.random.choice(self.all_actions)
        return action
    
    def learn(self):
        """learn from experience"""
        time.sleep(0.5)
        print("[INFO] The learning process complete. (ﾉ｀⊿´)ﾉ")
        return True
##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #