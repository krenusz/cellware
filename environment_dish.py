
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
ray.init(ignore_reinit_error=True)

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
        self.max_episode_length = 150
        self.state_observation = [self.x, self.y]
        self.reward = 0
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
        return [self.x, self.y]
    
    # Returns the number of actions in the action set
    def action_space(self):
        return self.actions
    # Agent takes the step, i.e. take action to interact with 
    #  the environment
    def step(self, action, current_state):
        # If agent is at terminal state, end the episode, set self.done to be True
        #if self.state_observation == [self.MAX_HOR_VAL, self.MAX_VER_VAL]:
        #    self.done = True
        #    return np.array(self.state_observation), self.reward, self.done, self.episode_length
        self.reward = 0
        if self.food1_collected == True and self.food2_collected == True and self.food3_collected == True:
            print('all the food collected')
            self.done = True
            self.all_food_collected = True
            return np.array(self.state_observation), self.reward, self.done, self.all_food_collected, self.food1_loc, self.food2_loc, self.food3_loc, self.episode_length, self.food1_collected, self.food2_collected, self.food3_collected
        
        elif self.episode_length > self.max_episode_length:
            self.done = True
            return np.array(self.state_observation), self.reward, self.done, self.all_food_collected, self.food1_loc, self.food2_loc, self.food3_loc, self.episode_length, self.food1_collected, self.food2_collected, self.food3_collected
        
        self.action = action
        self.current_state_observation = current_state
        self.state_observation = self.take_action()
        self.reward = self.get_reward()
        self.episode_length += 1
               
        return np.array(self.state_observation), self.reward, self.done, self.all_food_collected, self.food1_loc, self.food2_loc, self.food3_loc, self.episode_length, self.food1_collected, self.food2_collected, self.food3_collected
    
    def get_reward(self):

        
        
        # If agent tries to run out of the grid, penalize -10
        if all(self.current_state_observation == self.state_observation):
            reward = -10
   
        # If agent reaches the food, reward it with 100 
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

        return reward
    
    # Method to take action, remain in the same box if agent tries
    # to run outside the grid, otherwise move one box in the 
    # direction of the action
    def take_action(self):  
        #if self.x > -1 and self.x <= self.MAX_HOR_VAL:
        if (self.action == 0 and self.x == 0) or (self.action == 2 and self.x == self.MAX_HOR_VAL):
            self.x = self.x
        elif(self.action == 0):
            self.x -= 1
        elif(self.action == 2):
            self.x += 1
        else:
            self.x = self.x
            
        #if self.y > -1 and self.y <= self.MAX_VER_VAL:
        if (self.action == 1 and self.y == 0) or (self.action == 3 and self.y == self.MAX_HOR_VAL):
            self.y = self.y
        elif(self.action == 1):
            self.y -= 1
        elif(self.action == 3):
            self.y += 1
        else:
            self.y = self.y
                        
        return [self.x, self.y]
    
    # Method to align the environment between agents
    def align(self, food1_collected_align, food2_collected_align, food3_collected_align):
        self.food1_collected = food1_collected_align
        self.food2_collected = food2_collected_align
        self.food3_collected = food3_collected_align
        if self.food1_collected == True and self.food2_collected == True and self.food3_collected == True:
            self.all_food_collected = True
            
