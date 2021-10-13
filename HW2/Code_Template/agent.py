# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
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
    # add paras size of action space and learning rate
    def __init__(self, all_actions, num_actions, lr):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 1.0

        # initialize Q-table by the size of action space and state space (4*12)
        self.q_table = [[0 for _ in range(num_actions)] for _ in range(4*12)]

        # learning rate
        self.lr = lr

    def is_terminal(self,s):
        """determine whether s is in the terminal state """
        return s == 11

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        # # random strategy
        # action = np.random.choice(self.all_actions)

        # if s is not terminal
        if (not self.is_terminal(observation)):
            # calculate pi according to Q and exploration strategy
            action = 0

            max_q = 0

            for i in range(len(self.q_table[observation])):
                if (self.q_table[observation][i] > max_q):
                    max_q = self.q_table[observation][i]
                    action = i

        return action
    
    # add paras previous state, next state, reward and discounting factor
    def learn(self, s, s_, r, gamma):
        """learn from experience"""
        # time.sleep(0.5)

        max_q = 0

        for i in range(len(self.q_table[s_])):
            if (self.q_table[s_][i] > max_q):
                max_q = self.q_table[s_][i]

        for i in range(len(self.q_table[s_])):
            # Q-learning update rule
            self.q_table[s_][i] = (1-self.lr)*self.q_table[s][i] + self.lr*(r + gamma*max_q)

        print("[INFO] The learning process complete. (ﾉ｀⊿´)ﾉ")
        return True
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        do_something = True
        return None

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