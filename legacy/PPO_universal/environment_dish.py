
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from collections import namedtuple, deque
import random
import math
import matplotlib.pyplot as plt
import ray

@ray.remote
class Environment:

    # Constructor for GridWorld_Env Object, i.e. our agent
    def __init__(self, hor, ver):
        self.actions = [0, 1, 2, 3, 4] 
        self.MAX_HOR_VAL = hor-1
        self.MAX_VER_VAL = ver-1
        self.done = False
        self.episode_length = 0
        self.max_episode_length = 128
        self.reward = 0
        self.lvl_1_reward_table = np.array([ [1, 1, 1, 1, 2, 2, 3, 3, 0, 0],
                                             [1, 1, 1, 1, 2, 2, -1, 4, -1, -1],
                                             [1, 1, 1, 1, -1, -1, -1, 4, 50, 0],
                                             [1, 1, 1, 1, -1, -1, -1, 0, 0, 0],
                                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
                                             [0, 10, 20, 10, 0, 0, 20, -1, 60, 0],
                                             [10, 20, 30, 20, 10, 20, 30, -1, 6, 0],
                                             [20, 30, 90, 30, 20, 10, 20, -1, 6, -1],
                                             [10, 20, 30, 20, 10, 5, 5, -1, 0, 0],
                                             [0, 10, 20, 10, 0, 0, 0, 70, 0, 0]])
        
        
        self.lvl_2_reward_table = np.array([ [5, 5, 5, 5, 5, 5, 10, 20, 10, 5],
                                        [5, 5, 5, 5, 5, 10, 20, 30, 20, 10],
                                        [5, 5, 5, 5, 10, 20, 30, 40, 30, 20],
                                        [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
                                        [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
                                        [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
                                        [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
                                        [5, 5, 5, 10, 10, 20, 30, 40, 30, 20],
                                        [5, 5, 5, 5, 5, 10, 20, 30, 20, 10],
                                        [5, 5, 5, 5, 5, 5, 10, 20, 10, 5]])
    # Reset the agent at the start of each episode
    def reset(self):

        self.done = False
        self.episode_length = 0
        self.reward = 0
        self.lvl_1_reward_table = np.array([ [1, 1, 1, 1, 2, 2, 3, 3, 0, 0],
                                             [1, 1, 1, 1, 2, 2, -1, 4, -1, -1],
                                             [1, 1, 1, 1, -1, -1, -1, 4, 50, 0],
                                             [1, 1, 1, 1, -1, -1, -1, 0, 0, 0],
                                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
                                             [0, 10, 20, 10, 0, 0, 20, -1, 60, 0],
                                             [10, 20, 30, 20, 10, 20, 30, -1, 6, 0],
                                             [20, 30, 90, 30, 20, 10, 20, -1, 6, -1],
                                             [10, 20, 30, 20, 10, 5, 5, -1, 0, 0],
                                             [0, 10, 20, 10, 0, 0, 0, 70, 0, 0]])
        #self.lvl_1_reward_table = np.array([ [5, 5, 5, 5, 5, 10, 20, 10, 5, 5],
        #                                     [5, 5, 5, 5, 10, 20, 90, 20, 10, 5],
        #                                     [5, 5, 5, 5, 5, 10, 20, 10, 5, 5],
        #                                     [5, 5, 5, 5, 5, 5, 10, 5, 5, 5],
        #                                     [5, 5, 10, 5, 5, 5, 5, 20, 5, 5],
        #                                     [5, 10, 20, 10, 5, 5, 20, 30, 20, 5],
        #                                     [10, 20, 30, 20, 10, 20, 30, 90, 30, 20],
        #                                     [20, 30, 90, 30, 20, 10, 20, 30, 20, 5],
        #                                     [10, 20, 30, 20, 10, 5, 5, 20, 5, 5],
        #                                     [5, 10, 20, 10, 5, 5, 5, 5, 5, 5]])
        
        self.lvl_2_reward_table = np.array([ [5, 5, 5, 5, 5, 5, 10, 20, 10, 5],
                                        [5, 5, 5, 5, 5, 10, 20, 30, 20, 10],
                                        [5, 5, 5, 5, 10, 20, 30, 40, 30, 20],
                                        [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
                                        [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
                                        [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
                                        [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
                                        [5, 5, 5, 10, 10, 20, 30, 40, 30, 20],
                                        [5, 5, 5, 5, 5, 10, 20, 30, 20, 10],
                                        [5, 5, 5, 5, 5, 5, 10, 20, 10, 5]])
        
    
    # Returns the number of actions in the action set
    def action_space(self):
        return self.actions
  

    def step(self, action, current_state):
        
        state_observation = self.take_action(action, current_state)
        reward_lvl1, reward_lvl2 = self.get_reward(current_state, state_observation)
                
        return state_observation, reward_lvl1, reward_lvl2
    

    def get_reward(self, current_state, state_observation):
        
        if all(current_state == state_observation):
            reward_lvl1 = 0
            reward_lvl2 = 0
        else:
            reward_lvl1 = self.lvl_1_reward_table[state_observation[0], state_observation[2]] + self.lvl_1_reward_table[state_observation[1], state_observation[3]]
            reward_lvl2 = self.lvl_2_reward_table[state_observation[0], state_observation[2]] + self.lvl_2_reward_table[state_observation[1], state_observation[3]]

            copy = self.lvl_1_reward_table.copy()
            copy2 = self.lvl_2_reward_table.copy()
            
            copy[state_observation[0], state_observation[2]] = 0
            copy[state_observation[1], state_observation[3]] = 0
            copy2[state_observation[0], state_observation[2]] = 0
            copy2[state_observation[1], state_observation[3]] = 0

            self.lvl_1_reward_table = copy
            self.lvl_2_reward_table = copy2

        return reward_lvl1, reward_lvl2
    
    
    # Method to take action, remain in the same box if agent tries
    # to run outside the grid, otherwise move one box in the 
    # direction of the action
    def take_action(self, action, current_state):  
        state = current_state.copy()
        if action == 4 and state[2] == state[3] and (state[2] == self.MAX_HOR_VAL or self.lvl_1_reward_table[state[0], state[2]+1] == -1):
            state = state
            return state
        
        elif action == 4 and state[0] == state[1] and (state[0] == self.MAX_VER_VAL or self.lvl_1_reward_table[state[0]+1, state[2]] == -1):
            state = state
            return state
        
        elif action == 4 and state[2] == state[3]:
            state[1] = state[0]
            state[3] = state[2]+1
            return state
        
        elif action == 4 and state[0] == state[1]:
            state[1] = state[0]+1
            state[3] = state[2]
            return state


        
        
        if (action == 0 and any(state[0:2] == 0)) or (action == 2 and any(state[0:2] == self.MAX_VER_VAL)) or (action == 0 and ((self.lvl_1_reward_table[state[0]-1, state[2]] == -1) or (self.lvl_1_reward_table[state[1]-1, state[3]] == -1))) or (action == 2 and ((self.lvl_1_reward_table[state[0]+1, state[2]] == -1) or (self.lvl_1_reward_table[state[1]+1, state[3]] == -1))):
            state = state
        elif (action == 1 and any(state[2:4] == 0)) or (action == 3 and any(state[2:4] == self.MAX_HOR_VAL)) or (action == 1 and ((self.lvl_1_reward_table[state[0], state[2]-1] == -1) or (self.lvl_1_reward_table[state[1], state[3]-1] == -1))) or (action == 3 and ((self.lvl_1_reward_table[state[0], state[2]+1] == -1) or (self.lvl_1_reward_table[state[1], state[3]+1] == -1))):
            state = state
        elif(action == 0):
            state[0:2] = np.subtract(state[0:2],1)
        elif(action == 2):
            state[0:2] = np.add(state[0:2],1)

        elif(action == 1):
            state[2:4] = np.subtract(state[2:4],1)
        elif(action == 3):
            state[2:4] = np.add(state[2:4],1)
        else:
            state = state
                        
        return state
    
    # Method to align the environment
    def align(self, x, y):
        
            
        copy = self.lvl_1_reward_table.copy()
        copy2 = self.lvl_2_reward_table.copy()
        copy[x, y] = 0
        copy2[x, y] = 0

        self.lvl_1_reward_table = copy
        self.lvl_2_reward_table = copy2
    
    def get_attributes(self):
        return self.lvl_1_reward_table, self.lvl_2_reward_table, self.max_episode_length
    

