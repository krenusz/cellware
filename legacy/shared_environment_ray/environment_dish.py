
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
        self.actions = [0, 1, 2, 3] 
        self.x = 0
        self.y = 0
        self.MAX_HOR_VAL = hor-1
        self.MAX_VER_VAL = ver-1
        self.food1_loc = [0, 4]
        self.food2_loc = [4, 0]
        self.food3_loc = [self.MAX_HOR_VAL, self.MAX_VER_VAL]
        self.done = False
        self.all_food_collected = False
        self.food1_collected = False
        self.food2_collected = False
        self.food3_collected = False
        self.episode_length = 0
        self.max_episode_length = 128
        self.state_observation = [self.x, self.y]
        self.reward = 0
        self.lvl_1_reward_table = np.array([ [1, 1, 1, 1, 1, 10, 20, 10, 1, 1],
                                             [1, 1, 1, 1, 10, 20, 90, 20, 10, 1],
                                             [1, 1, 1, 1, 1, 10, 20, 10, 1, 1],
                                             [1, 1, 1, 1, 1, 1, 10, 1, 1, 1],
                                             [1, 1, 10, 1, 1, 1, 1, 20, 1, 1],
                                             [1, 10, 20, 10, 1, 1, 20, 30, 20, 1],
                                             [10, 20, 30, 20, 10, 20, 30, 90, 30, 20],
                                             [20, 30, 90, 30, 20, 10, 20, 30, 20, 1],
                                             [10, 20, 30, 20, 10, 1, 1, 20, 1, 1],
                                             [1, 10, 20, 10, 1, 1, 1, 1, 1, 1]])
        
        
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
        self.all_food_collected = False
        self.food1_collected = False
        self.food2_collected = False
        self.food3_collected = False
        self.episode_length = 0
        self.x, self.y = 0, 0
        self.state_observation = [self.x, self.y]
        self.reward = 0
        self.lvl_1_reward_table = np.array([ [1, 1, 1, 1, 1, 10, 20, 10, 1, 1],
                                             [1, 1, 1, 1, 10, 20, 90, 20, 10, 1],
                                             [1, 1, 1, 1, 1, 10, 20, 10, 1, 1],
                                             [1, 1, 1, 1, 1, 1, 10, 1, 1, 1],
                                             [1, 1, 10, 1, 1, 1, 1, 20, 1, 1],
                                             [1, 10, 20, 10, 1, 1, 20, 30, 20, 1],
                                             [10, 20, 30, 20, 10, 20, 30, 90, 30, 20],
                                             [20, 30, 90, 30, 20, 10, 20, 30, 20, 1],
                                             [10, 20, 30, 20, 10, 1, 1, 20, 1, 1],
                                             [1, 10, 20, 10, 1, 1, 1, 1, 1, 1]])
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
        return [self.x, self.y]
    
    # Returns the number of actions in the action set
    def action_space(self):
        return self.actions
    
    #def reset_reward_map(self):
        
        

    # Agent takes the step, i.e. take action to interact with 
    #  the environment
    def step(self, action, current_state):
        
        state_observation = self.take_action(action, current_state)
        reward_lvl1, reward_lvl2 = self.get_reward(current_state, state_observation)
                
        return np.array(state_observation), reward_lvl1, reward_lvl2
    
    def get_reward1(self, current_state, state_observation):
        x = state_observation[0]
        y = state_observation[1]
        reward_lvl1 = self.lvl_1_reward_table[x, y]
        reward_lvl2 = self.lvl_2_reward_table[x, y]

        if all(current_state == state_observation):
            reward_lvl1 = 0
            reward_lvl2 = 0

        copy = self.lvl_1_reward_table.copy()
        copy2 = self.lvl_2_reward_table.copy()
        if copy[x, y] > 5:
            copy[x, y] -= 1
        if copy2[x, y] > 5:
            copy2[x, y] -= 1

        self.lvl_1_reward_table = copy
        self.lvl_2_reward_table = copy2

        return reward_lvl1, reward_lvl2
    
    def get_reward(self, current_state, state_observation):
        x = state_observation[0]
        y = state_observation[1]
        reward_lvl1 = self.lvl_1_reward_table[x, y]
        reward_lvl2 = self.lvl_2_reward_table[x, y]

        if all(current_state == state_observation):
            reward_lvl1 = 0
            reward_lvl2 = 0

        if reward_lvl1 == 90 or reward_lvl2 == 90:
            copy = self.lvl_1_reward_table.copy()
            copy2 = self.lvl_2_reward_table.copy()
            
            copy[x, y] = 1
            copy2[x, y] = 1

            self.lvl_1_reward_table = copy
            self.lvl_2_reward_table = copy2
        else:
            copy = self.lvl_1_reward_table.copy()
            copy2 = self.lvl_2_reward_table.copy()
            
            copy[x, y] = 1
            copy2[x, y] = 1

            self.lvl_1_reward_table = copy
            self.lvl_2_reward_table = copy2

        return reward_lvl1, reward_lvl2
    
    
    # Method to take action, remain in the same box if agent tries
    # to run outside the grid, otherwise move one box in the 
    # direction of the action
    def take_action(self, action, current_state):  
        x = current_state[0]
        y = current_state[1]
        #if self.x > -1 and self.x <= self.MAX_HOR_VAL:
        if (action == 0 and x == 0) or (action == 2 and x == self.MAX_HOR_VAL):
            x = x
        elif(action == 0):
            x -= 1
        elif(action == 2):
            x += 1
        else:
            x = x
            
        #if self.y > -1 and self.y <= self.MAX_VER_VAL:
        if (action == 1 and y == 0) or (action == 3 and y == self.MAX_HOR_VAL):
            y = y
        elif(action == 1):
            y -= 1
        elif(action == 3):
            y += 1
        else:
            y = y
                        
        return [x, y]
    
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
    




def get_reward_ooo(self):    
        # If agent tries to run out of the grid, penalize -2
        if all(self.current_state_observation == self.state_observation):
            reward = -10
        
        # If agent reached Terminal state, reward = 0
        #elif (self.x, self.y) == (self.MAX_HOR_VAL-1, self.MAX_VER_VAL) and self.action == 2 and (self.food1_collected == True or self.food2_collected == True):
        #    reward = 5
        #    print('only 1 food collected')
        #elif (self.x, self.y) == (self.MAX_HOR_VAL, self.MAX_VER_VAL-1) and self.action == 3 and (self.food1_collected == True or self.food2_collected == True):
        #    reward = 5
        #    print('only 1 food collected')
        # For all other states, penalize agent with -1
        
        #if np.sum(self.current_state_observation) >= np.sum(self.state_observation):
        #   reward += -5
        # If agent reaches the food, reward it with 5 at the end of the episode
        elif (self.x, self.y) == (self.food1_loc[0], self.food1_loc[1]) and self.food1_collected == False:
            print('food1 collected')
            self.food1_collected = True
            reward = 100

        elif (self.x, self.y) == (self.food2_loc[0], self.food2_loc[1]) and self.food2_collected == False:
            print('food2 collected')
            self.food2_collected = True
            reward = 100

        elif (self.x, self.y) == (self.food3_loc[0], self.food3_loc[1]) and self.food3_collected == False:
            print('food3 collected')
            self.food3_collected = True
            reward = 100
        else:
            reward = -1

        #if self.food1_collected == True and self.food2_collected == True and self.food3_collected == True:
        #    reward = 200
        return reward
def step_ooo(self, action, current_state):
        # If agent is at terminal state, end the episode, set self.done to be True
        #if self.state_observation == [self.MAX_HOR_VAL, self.MAX_VER_VAL]:
        #    self.done = True
        #    return np.array(self.state_observation), self.reward, self.done, self.episode_length
        self.reward_lvl1 = 0
        self.reward_lvl2 = 0
        if self.food1_collected == True and self.food2_collected == True and self.food3_collected == True:
            print('all the food collected')
            self.done = True
            self.all_food_collected = True
            return np.array(self.state_observation), self.reward_lvl1, self.reward_lvl2
        
        elif self.episode_length > self.max_episode_length:
            self.done = True
            return np.array(self.state_observation), self.reward_lvl1, self.reward_lvl2
        
        self.action = action
        self.current_state_observation = current_state
        self.state_observation = self.take_action()
        self.reward_lvl1, self.reward_lvl2 = self.get_reward()
        #if self.reward_lvl1 == -10 or self.reward_lvl2 == -10:
        #    self.state_observation = self.current_state_observation
        self.episode_length += 1