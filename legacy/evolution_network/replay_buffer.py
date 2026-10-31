import torch
import numpy as np

import copy


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.use_adv_norm = args.use_adv_norm
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.max_episode_steps
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.count = 0
        self.buffer = None
        self.reset_buffer()
        self.episode_step = 0

    def reset_buffer(self):
        self.buffer = {'s': np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
                       'v': np.zeros([self.batch_size, self.episode_limit + 1]),
                       'h': np.zeros([self.batch_size, self.episode_limit]),
                       'a': np.zeros([self.batch_size, self.episode_limit]),
                       'a_logprob': np.zeros([self.batch_size, self.episode_limit]),
                       'r': np.zeros([self.batch_size, self.episode_limit]),
                       'res': np.zeros([self.batch_size, self.episode_limit]),
                       'c': np.zeros([self.batch_size, self.episode_limit]),
                       'dw': np.ones([self.batch_size, self.episode_limit]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.batch_size, self.episode_limit])
                       }
        self.count = 0
        self.episode_num = 0
        self.episode_step = 0
        self.max_episode_len = 0


    def store_transition(self, s, v, h, a, a_logprob, r, res, c, dw):
        self.buffer['s'][self.episode_num][self.episode_step] = s
        self.buffer['v'][self.episode_num][self.episode_step] = v
        self.buffer['h'][self.episode_num][self.episode_step] = h
        self.buffer['a'][self.episode_num][self.episode_step] = a
        self.buffer['a_logprob'][self.episode_num][self.episode_step] = a_logprob
        self.buffer['r'][self.episode_num][self.episode_step] = r
        self.buffer['res'][self.episode_num][self.episode_step] = res
        self.buffer['c'][self.episode_num][self.episode_step] = c
        self.buffer['dw'][self.episode_num][self.episode_step] = dw

        self.buffer['active'][self.episode_num][self.episode_step] = 1.0
        self.count += 1
        self.episode_step += 1

    def replace(self, input_):
        for i,key in enumerate(self.buffer.keys()):
            self.buffer[key] = input_[i]
        self.episode_num = self.buffer['s'].shape[0]
        self.max_episode_len = self.buffer['s'].shape[1]
        self.count = self.episode_num * self.max_episode_len
        self.episode_step = self.buffer['s'].shape[1]

    def store_last_value(self, v):
        self.buffer['v'][self.episode_num][self.episode_step] = v
        self.episode_num += 1
        # Record max_episode_len
        if self.episode_step > self.max_episode_len:
            self.max_episode_len = self.episode_step
        self.episode_step = 0

    def get_adv(self):
        # Calculate the advantage using GAE
        v = self.buffer['v'][:, :self.max_episode_len]
        v_next = self.buffer['v'][:, 1:self.max_episode_len + 1]
        r = self.buffer['r'][:, :self.max_episode_len]
        dw = self.buffer['dw'][:, :self.max_episode_len]
        active = self.buffer['active'][:, :self.max_episode_len]
        adv = np.zeros_like(r)  # adv.shape=(batch_size,max_episode_len)
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len)
            deltas = r + self.gamma * v_next * (1 - dw) - v
            for t in reversed(range(self.max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae  # gae.shape=(batch_size)
                adv[:, t] = gae
            v_target = adv + v  # v_target.shape(batch_size,max_episode_len)
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv_copy = copy.deepcopy(adv)
                adv_copy[active == 0] = np.nan 
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
        return adv, v_target

    def get_training_data(self):
        adv, v_target = self.get_adv()
        batch = {'s': torch.tensor(self.buffer['s'], dtype=torch.float32),
                    'v': torch.tensor(self.buffer['v'], dtype=torch.float32),
                    'h': torch.tensor(self.buffer['h'], dtype=torch.float32),
                 'a': torch.tensor(self.buffer['a'], dtype=torch.long), 
                 'a_logprob': torch.tensor(self.buffer['a_logprob'], dtype=torch.float32),
                    'r': torch.tensor(self.buffer['r'], dtype=torch.float32),
                    'res': torch.tensor(self.buffer['res'], dtype=torch.float32),
                    'c': torch.tensor(self.buffer['c'], dtype=torch.float32),
                 'active': torch.tensor(self.buffer['active'], dtype=torch.float32),
                 'adv': torch.tensor(adv, dtype=torch.float32),
                 'v_target': torch.tensor(v_target, dtype=torch.float32)}

        return batch
    
    def numpy_to_tensor(self):
        batch = {'s': torch.tensor(self.buffer['s'], dtype=torch.float32),
                    'v': torch.tensor(self.buffer['v'], dtype=torch.float32),
                    'h': torch.tensor(self.buffer['h'], dtype=torch.float32),
                 'a': torch.tensor(self.buffer['a'], dtype=torch.long), 
                 'a_logprob': torch.tensor(self.buffer['a_logprob'], dtype=torch.float32),
                    'r': torch.tensor(self.buffer['r'], dtype=torch.float32),
                    'res': torch.tensor(self.buffer['res'], dtype=torch.float32),
                    'c': torch.tensor(self.buffer['c'], dtype=torch.float32),
                 'active': torch.tensor(self.buffer['active'], dtype=torch.float32)}
        return batch
    
    def numpy_(self):
        return self.buffer