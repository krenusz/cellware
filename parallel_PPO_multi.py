import argparse
import pickle
from collections import namedtuple
from itertools import count

from threadpoolctl import threadpool_limits
from billiard.pool import Pool
from multiprocessing import freeze_support
import ray
ray.init()

import os, time
import numpy as np
import matplotlib.pyplot as plt
from environment_dish import Environment
from cell import cells

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
#from tensorboardX import SummaryWriter

# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10
pool = Pool(4)
env = Environment(10,10)
num_state = len(env.reset())
num_action = len(env.action_space())
torch.manual_seed(seed)

#env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 20
    buffer_capacity = 1000
    batch_size = 100

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        #self.writer = SummaryWriter('../exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        #if not os.path.exists('../param'):
        #    os.makedirs('../param/net_param')
        #    os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1


    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
        total_action_loss = 0
        total_value_loss = 0
        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        print('Reward: ', Gt.sum())
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                #print(action_loss)
                #self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                total_action_loss += action_loss
                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                #self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                total_value_loss += value_loss
                self.training_step += 1
        print('Total Action Loss:',total_action_loss, 'Total Value Loss:',total_value_loss)
        del self.buffer[:] # clear experience
#from threadpoolctl import threadpool_limits
#from parallel_PPO_multi import Transition
#env = Environment
def cells1(agent, state, i_epoch):
    with threadpool_limits(limits=1, user_api='blas'):
        
        action, action_prob = agent.select_action(state)
        print('action',action,'state',state)

        next_state, reward, done, all_food_collected, food1_loc, food2_loc, food3_loc, _ = env.step(action,state)
        trans = Transition(state, action, action_prob, reward, next_state)
        if all(next_state == food1_loc) or all(next_state == food2_loc) or all(next_state == food3_loc):
            score = 1
        else:
            score = 0
        agent.store_transition(trans)

        if done:
            
            if len(agent.buffer) >= agent.batch_size:
                agent.update(i_epoch)

    return agent, next_state, score
def main():
    
    agent_list = [PPO()]
    state_list = [0]
    for i_epoch in range(10000):
        print('i_epoch:',i_epoch)
        for s in range(len(state_list)):
            state_list[s] = np.array(env.reset())
        #if len(agent_list) <= 4:
        #    pool = Pool(len(agent_list))
        #else:
        #    pool = Pool(4)
        #print('pool created',pool)

        a = torch.zeros(size=(env.MAX_HOR_VAL+1,env.MAX_VER_VAL+1))
        a[env.food1_loc[0],env.food1_loc[1]] = -200
        a[env.food2_loc[0],env.food2_loc[1]] = -200
        a[env.food3_loc[0],env.food3_loc[1]] = -200
        if render: 
            env.render()

        
        print('itt a baj?')
        for cnt in count():
            print('while loop started')
            if len(agent_list) <= 4:
                pool = Pool(len(agent_list))
            else:
                pool = Pool(4)

            print('pool created')
            print(cells1(agent_list[0], state_list[0], i_epoch))
            print('went through')
            output = pool.starmap(cells1, zip(agent_list, state_list, [i_epoch * len(agent_list)]))
            print('output created')
            pool.close()
        
            pool.join()
            
            
            agent = {k:v for listitem in output for (k,v) in listitem[0].items()}
            next_state = {k:v for listitem in output for (k,v) in listitem[1].items()}
            score = {k:v for listitem in output for (k,v) in listitem[2].items()}
            #Transition = {k:v for listitem in output for (k,v) in listitem[3].items()}
            #env = {k:v for listitem in output for (k,v) in listitem[4].items()}
            #done = [k:v for listitem in output for (k,v) in listitem[2].items()]
            #all_food_collected = [k:v for listitem in output for (k,v) in listitem[3].items()]
            
            
            score_list = [0]*len(agent_list)
                        
            for c in range(len(agent_list)):
                agent_list[c] = agent[c]
                state_list[c] = next_state[c]
                score_list[c] += score[c]
                a[next_state[c][0],next_state[c][1]] += 1

            best_score_index = score_list.index(max(score_list))     
            
            if env.all_food_collected:
                print('All food collected! next agent joining...','collecter: Agent',c)
                
                agent_list.append(agent[best_score_index])
                state_list.append(next_state[best_score_index])
                break
            if env.done:
                
                break
                    
       
                
                
            #if done or all_food_collected:
            #    print('Number of agents in play:',len(agent_list))
            #    time.sleep(0.5)
            #    break       
         
        print(a)

if __name__ == '__main__':
    #    freeze_support()
    main()
    print("end")