import torch
import numpy as np

import copy


class ReplayBuffer_reproducer:
    def __init__(self, args):
        self.args = args
        self.state_dim = args.state_dim_reprod
        self.action_dim = args.action_dim_reprod
        self.episode_limit = args.reprod_mini_batch_size
        self.batch_size = args.reprod_batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.count = 0
        self.buffer = None
        self.reset_buffer()
        self.episode_step = 0

    def reset_buffer(self):
        self.buffer = {'s': np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
                       'a': np.zeros([self.batch_size, self.episode_limit]),
                       'a_logprob': np.zeros([self.batch_size, self.episode_limit]),
                       'r': np.zeros([self.batch_size, self.episode_limit]),
                       'res': np.zeros([self.batch_size, self.episode_limit]),
                       'l': np.zeros([self.batch_size, self.episode_limit]),
                       'c': np.zeros([self.batch_size, self.episode_limit]),
                       'active': np.zeros([self.batch_size, self.episode_limit])
                       }
        self.count = 0
        self.episode_num = 0
        self.episode_step = 0
        self.max_episode_len = 0


    def store_transition(self, s, a, a_logprob, r, res, l, c):
        self.buffer['s'][self.episode_num][self.episode_step] = s
        self.buffer['a'][self.episode_num][self.episode_step] = a
        self.buffer['a_logprob'][self.episode_num][self.episode_step] = a_logprob
        self.buffer['r'][self.episode_num][self.episode_step] = r
        self.buffer['res'][self.episode_num][self.episode_step] = res
        self.buffer['l'][self.episode_num][self.episode_step] = l
        self.buffer['c'][self.episode_num][self.episode_step] = c
        
        self.buffer['active'][self.episode_num][self.episode_step] = 1.0
        self.count += 1
        self.episode_step += 1

        if self.episode_step == self.episode_limit:
            self.episode_num += 1
            if self.episode_step > self.max_episode_len:
                self.max_episode_len = self.episode_step
            self.episode_step = 0


    def get_training_data(self):        
        batch = {'s': torch.tensor(self.buffer['s'], dtype=torch.float32),
                 'a': torch.tensor(self.buffer['a'], dtype=torch.long), 
                 'a_logprob': torch.tensor(self.buffer['a_logprob'], dtype=torch.float32),
                    'r': torch.tensor(self.buffer['r'], dtype=torch.float32),
                    'res': torch.tensor(self.buffer['res'], dtype=torch.float32),
                    'l': torch.tensor(self.buffer['l'], dtype=torch.float32),
                    'c': torch.tensor(self.buffer['c'], dtype=torch.float32),
                 'active': torch.tensor(self.buffer['active'], dtype=torch.float32)}
        return batch
    
    def numpy_to_tensor(self):
        batch = {'s': torch.tensor(self.buffer['s'], dtype=torch.float32),
                 'a': torch.tensor(self.buffer['a'], dtype=torch.long), 
                 'a_logprob': torch.tensor(self.buffer['a_logprob'], dtype=torch.float32),
                    'r': torch.tensor(self.buffer['r'], dtype=torch.float32),
                    'res': torch.tensor(self.buffer['res'], dtype=torch.float32),
                    'l': torch.tensor(self.buffer['l'], dtype=torch.float32),
                    'c': torch.tensor(self.buffer['c'], dtype=torch.float32),
                 'active': torch.tensor(self.buffer['active'], dtype=torch.float32)}
        return batch
    
    def numpy_(self):
        return self.buffer