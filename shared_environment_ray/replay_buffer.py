import torch
import numpy as np
import pandas as pd


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.count = 0

    
        
    def store(self, s, a, a_logprob, r, s_):
        s_copy = self.s.copy()
        a_copy = self.a.copy()
        a_logprob_copy = self.a_logprob.copy()
        r_copy = self.r.copy()
        s__copy = self.s_.copy()

        s_copy[self.count] = s
        a_copy[self.count] = a
        a_logprob_copy[self.count] = a_logprob
        r_copy[self.count] = r
        s__copy[self.count] = s_

        self.s = s_copy
        self.a = a_copy
        self.a_logprob = a_logprob_copy
        self.r = r_copy
        self.s_ = s__copy

        self.count += 1
    
    def reset(self):
        self.s = np.zeros((self.args.batch_size, self.args.state_dim))
        self.a = np.zeros((self.args.batch_size, self.args.action_dim))
        self.a_logprob = np.zeros((self.args.batch_size, self.args.action_dim))
        self.r = np.zeros((self.args.batch_size, 1))
        self.s_ = np.zeros((self.args.batch_size, self.args.state_dim))
        self.count = 0
    
    def append(self, input_):
        s, a, a_logprob, r, s_ = input_    
        self.s = np.append(self.s, s, axis=0)
        self.a = np.append(self.a, a, axis=0)
        self.a_logprob = np.append(self.a_logprob, a_logprob, axis=0)
        self.r = np.append(self.r, r, axis=0)
        self.s_ = np.append(self.s_, s_, axis=0)
        self.count += len(s)

    def replace(self, input_):
        s, a, a_logprob, r, s_ = input_    
        self.s = s
        self.a = a
        self.a_logprob = a_logprob
        self.r = r
        self.s_ = s_
        self.count = len(s)

    def numpy_(self):
        return self.s, self.a, self.a_logprob, self.r, self.s_
    
    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)

        return s, a, a_logprob, r, s_
    
    def save_memory(self, path):
        memory = self.numpy_()
        df = pd.DataFrame(index=range(self.count),columns=['state', 'actions', 'action_prob', 'reward', 'next_state'])
        for i,col in enumerate(df.columns):
            if memory[i].shape[1] == 1:
                df[col] = memory[i][:self.count]
            else:
                df[col] = list(memory[i])[:self.count]
        df.to_csv('{}_memory.csv'.format(path))
