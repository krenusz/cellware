from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
import torch.nn as nn
from torch.distributions import Beta, Normal, Categorical
from replay_buffer import ReplayBuffer
from replay_buffer_reproducer import ReplayBuffer_reproducer
import copy
import ray

from policies import Actor_Discrete, Actor_RNN, Actor_Beta, Actor_Gaussian, Critic, Critic_RNN, RNN_Encoder, RNN_reprod

@ray.remote
class PPO_universal_selfreplicate():
    def __init__(self, args, env, agent_id, policy_dist):
        self.agent_id = agent_id
        self.env = env
        self.replay_buffer = copy.copy(ReplayBuffer(args))
        self.replay_buffer_reproducer = copy.copy(ReplayBuffer_reproducer(args))
        #self.replay_buffer = ReplayBuffer(args)  # Create replay buffer
        self.policy_dist = policy_dist
        self.max_action = args.max_action
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.reprod_batch_size = args.reprod_batch_size
        self.reprod_mini_batch_size = args.reprod_mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.max_episode_steps = args.max_episode_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.lr_r = args.lr_r  # Learning rate of reproducer
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_collective = args.use_collective
        self.collective_switch = args.collective_switch
        self.total_actor_loss = 0
        self.total_critic_loss = 0
        self.total_advantage = 0
        self.total_entropy_loss = 0
        self.target_kl = args.target_kl
        self.stop_training = False
        #self.writer = args.writer
        self.done = False
        self.score = 0
        self.env_name = args.env_name
        self.lvl = args.lvl
        self.agent_number = args.agent_number
        self.use_shuffle = args.use_shuffle
        
        if self.policy_dist == "Discrete":
            self.actor = Actor_Discrete(args)
            self.critic = Critic(args)
        elif self.policy_dist == "Discrete RNN":
            self.actor = Actor_RNN(args)
            self.critic = Critic_RNN(args)
        elif self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
            self.critic = Critic(args)
        else:
            self.actor = Actor_Gaussian(args)
            self.critic = Critic(args)
        
        self.encoder = RNN_Encoder(args, 6, 1, 1, 0.2)
        
        self.reproducer = RNN_reprod(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5, maximize=False)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
            self.optimizer_encoder = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
            self.optimizer_reproducer = torch.optim.Adam(self.reproducer.parameters(), lr=self.lr_r, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
            self.optimizer_encoder = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
            self.optimizer_reproducer = torch.optim.Adam(self.reproducer.parameters(), lr=self.lr_r)

    def achievement_reset(self):
        self.done = False
        self.score = 0

    def reset_rnn_hidden(self):
        self.actor.reset_rnn_hidden()
        self.critic.reset_rnn_hidden()

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        state = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)

        s, a, _, _, res, _, _  = self.replay_buffer.numpy_to_tensor()
        if self.policy_dist == "Beta":
            a = self.actor.mean(state).detach().numpy().flatten()
        else:
            a = self.actor(state).detach().numpy().flatten()
        return a

    def get_value(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
            value = self.critic(s)
            return value.item()

    def choose_action(self, state, i_epoch, evaluate=False):
        episode_step = self.replay_buffer.episode_step
        episode_num = self.replay_buffer.episode_num
        if episode_step < 6: 
            episode_step = 6

        
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)

        batch = self.replay_buffer.numpy_to_tensor()
        s,a,res = batch['s'], batch['a'], batch['res']
        
        
        enc_input = torch.cat((batch['s'],batch['a'].view(self.batch_size,self.mini_batch_size,1),batch['res'].view(self.batch_size,self.mini_batch_size,1)),dim=-1)[episode_num,episode_step-6:episode_step,:]
        
        with torch.no_grad():
            h_end = self.encoder(enc_input)
        h_end = torch.unsqueeze(torch.tensor(h_end, dtype=torch.float), 0)

        final = torch.cat((state,h_end),dim=1)
        #print('final',final.shape, 'episode', i_epoch)
        if self.policy_dist == "Discrete":
            with torch.no_grad():
                dist = self.actor.get_dist(final)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
        elif self.policy_dist == "Discrete RNN":
            with torch.no_grad():
                a, a_logprob, logit = self.actor.get_dist(final)
                if evaluate:
                    a = torch.argmax(logit)
                    return a.item(), None
        elif self.policy_dist == "Gaussian":        
            with torch.no_grad():
                dist = self.actor.get_dist(final)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, 0, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)            
        elif self.policy_dist == "Beta":
            raise ValueError('Beta distribution not supported yet')
            with torch.no_grad():
                dist = self.actor.get_dist(final)
                
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            raise ValueError('Policy distribution not recognized')
                
        return a.numpy().flatten(), a_logprob.numpy().flatten(), h_end.numpy().flatten()
    

    def update(self, total_steps, replay_buffer = 0):
            
        if replay_buffer == 0:
            replay_buffer = self.replay_buffer
        
        batch = replay_buffer.get_training_data()

        
        #self.total_advantage = sum(adv).item()
        approx_kl_divs = []
        self.total_actor_loss = 0
        self.total_critic_loss = 0
        self.total_advantage = 0
        self.total_entropy_loss = 0
        # Optimize policy for K epochs:
        if self.use_shuffle:
            shuffle_times = self.K_epochs
            sampler = SubsetRandomSampler(range(self.batch_size))
        else:
            shuffle_times = self.K_epochs
            sampler = SequentialSampler(range(self.batch_size))
        for _ in range(shuffle_times):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(sampler, self.mini_batch_size, False):
                #print('Memory', batch['s'].shape,batch['h'].shape)
                #print(index)
                #if 0 in index:
                #    av_resource = torch.cat((torch.tensor([[1]]),res[[i-1 for i in index if i > 0]]))
                #else:
                #    av_resource = res[np.array(index)-1]

                # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                if self.policy_dist == "Discrete RNN":
                    self.reset_rnn_hidden()
                #print(batch['s'][index].shape,batch['h'][index].shape)
                 # logits_now.shape=(mini_batch_size, max_episode_len, action_dim)
                values_now = self.critic.forward(torch.cat((batch['s'][index],batch['h'][index].view(self.batch_size,self.mini_batch_size,1)),dim=-1)).squeeze(-1)  # values_now.shape=(mini_batch_size, max_episode_len)
                
                if self.policy_dist == "Discrete RNN":
                    now = self.actor.forward(torch.cat((batch['s'][index],batch['h'][index].view(self.batch_size,self.mini_batch_size,1)),dim=-1)) 
                    dist_now = Categorical(logits=now)
                elif self.policy_dist == "Gaussian":
                    dist_now = self.actor.get_dist(torch.cat((batch['s'][index],batch['h'][index].view(self.batch_size,self.mini_batch_size,1)),dim=-1))
                elif self.policy_dist == "Discrete":
                    now = self.actor.forward(torch.cat((batch['s'][index],batch['h'][index].view(self.batch_size,self.mini_batch_size,1)),dim=-1)) 
                    dist_now = Categorical(probs=now)
                else:
                    raise ValueError('Policy distribution not recognized (Beta distribution not supported yet)')

                
                if self.policy_dist == "Gaussian":
                    dist_entropy = dist_now.entropy().squeeze(dim=-1) # shape(mini_batch_size, max_episode_len)
                    a_logprob_now = dist_now.log_prob(batch['a'][index].view(self.batch_size,self.mini_batch_size,1)).squeeze(dim=-1)
                    ratios = torch.exp(a_logprob_now - batch['a_logprob'][index]) 
                else:
                    dist_entropy = dist_now.entropy() # shape(mini_batch_size, max_episode_len)
                    a_logprob_now = dist_now.log_prob(batch['a'][index])  # shape(mini_batch_size, max_episode_len)
                    ratios = torch.exp(a_logprob_now - batch['a_logprob'][index])
                # a/b=exp(log(a)-log(b))
                 # shape(mini_batch_size, max_episode_len)
                print(ratios.shape)
                # actor loss
                surr1 = ratios * batch['adv'][index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch['adv'][index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size, max_episode_len)
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                # critic_loss
                critic_loss = (values_now - batch['r'][index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = a_logprob_now - batch['a_logprob'][index]
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).numpy()
                    approx_kl_divs.append(approx_kl_div)

                
                
                # Update
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                self.optimizer_encoder.zero_grad()
                loss = actor_loss + critic_loss * 0.5
                loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                self.optimizer_encoder.step()   
                self.total_actor_loss += actor_loss
                self.total_critic_loss += critic_loss
                self.total_entropy_loss += dist_entropy.mean()
                self.total_advantage += batch['adv']
                print('LOSS/K_epoch:',loss, 'Total Reward:', batch['r'][index].sum(), 'Total Resource:', batch['res'][index].sum(), 'Total Adv:', batch['adv'][index].sum(), 'Total Pred Reward:', batch['v_target'][index].sum())
        explained_var = self.explained_variance(np.asarray(batch['v'][index,:-1]).flatten(), np.asarray(batch['r'][index]).flatten())
        print('Explained Variance:',explained_var, 'Mean KL Divergence:', np.mean(approx_kl_divs))
        
        if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
            self.stop_training = True
            print(f"Early stopping at step {total_steps} due to reaching max kl: {approx_kl_div:.2f}")
                
        if self.policy_dist == "Discrete RNN":
            self.reset_rnn_hidden()
        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)
    
        
    def act(self, state, i_epoch, lvl):
        
        from threadpoolctl import threadpool_limits
    
        
        with threadpool_limits(limits=1, user_api='blas'):
            score = 0
            
            if self.done:
                print('done', 'agent', self.agent_id)
                return state, self.done
            
            if self.use_collective:
                if self.replay_buffer.episode_num == self.batch_size and self.replay_buffer.count == self.mini_batch_size*self.batch_size:
                
                    self.update(i_epoch)
                    self.replay_buffer.reset_buffer()
                      
                        
            a, action_prob, h_end = self.choose_action(state, i_epoch)

            value = self.get_value(torch.cat((torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0),torch.unsqueeze(torch.tensor(h_end, dtype=torch.float), 0)),dim=1))
            
            a = a.round()
            if self.policy_dist == "Beta":
                action = 2 * (a - 0.5) * max(self.env.actions)  # [0,1]->[-max,max]
            else:  
                action = a
            #print( 'policy',self.policy_dist,'action',action, 'agent', self.agent_id)
            if action > 3:
                cost = 2
            else:
                cost = 1
            
            next_state, reward_lvl1, reward_lvl2 = ray.get(self.env.step.remote(action,state))
            
            if lvl == 1:
                reward = reward_lvl1
            else:
                reward = reward_lvl2
            
            if reward == 90 and lvl == 1:
                self.score += 1
                
            elif reward == 90 and lvl == 2:
                self.done = True
                self.update(i_epoch)
                print('place taken by', self.agent_id)

            elif reward != 90 and lvl == 1:
                self.score = 0
                
            elif reward != 90 and lvl == 2:
                self.done = False
            
            if self.replay_buffer.episode_step == self.mini_batch_size:
                v_ = self.get_value(torch.cat((torch.unsqueeze(torch.tensor(next_state, dtype=torch.float), 0),torch.unsqueeze(torch.tensor(h_end, dtype=torch.float), 0)),dim=1))
                self.replay_buffer.store_last_value(v_)
                dw = True
            else:
                dw = False
            
            self.replay_buffer.store_transition(state, value, h_end, action, action_prob, reward, reward-cost, cost, dw)
            
            if self.replay_buffer.episode_num == self.batch_size-1 and self.replay_buffer.count == self.mini_batch_size*self.batch_size - 1:
                
                self.update(i_epoch)
                self.replay_buffer.reset_buffer()

        if lvl == 1:
            return next_state, self.score
        else:
            return next_state, self.done
    
    def select_differentiation_action(self):
        from threadpoolctl import threadpool_limits
    
        state = self.policy_mapping()
        with threadpool_limits(limits=1, user_api='blas'):
            with torch.no_grad():
                a, a_logprob, logit = self.reproducer.get_dist(torch.tensor([state], dtype=torch.float32).unsqueeze(dim=0))

            if state == a:
                cost=1
            else:
                cost=2

            return state, a, a_logprob, cost
    
    def differentiate(self, args, action, agent_id):
        old_policy = self.policy_dist
        policy = self.policy_decoding(action)
        self.policy_dist = policy
        self.agent_id = agent_id

        if old_policy != policy:
            if self.policy_dist == "Discrete":
                self.actor = Actor_Discrete(args)
                self.critic = Critic(args)
            elif self.policy_dist == "Discrete RNN":
                self.actor = Actor_RNN(args)
                self.critic = Critic_RNN(args)
            elif self.policy_dist == "Beta":
                self.actor = Actor_Beta(args)
                self.critic = Critic(args)
            else:
                self.actor = Actor_Gaussian(args)
                self.critic = Critic(args)
            
            if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
                self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5, maximize=False)
                self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
                
            else:
                self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
                self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
                


    def reproduction_rewarding(self, state, action, action_prob, reward, loss, cost ,i_evolution):
        
        self.replay_buffer_reproducer.store_transition(state, action, action_prob, reward, reward-cost, loss, cost)
        print('count', self.replay_buffer_reproducer.count, 'episode', self.replay_buffer_reproducer.episode_num, 'i_evolution', i_evolution)
        if self.replay_buffer_reproducer.episode_num == self.reprod_batch_size-1 and self.replay_buffer_reproducer.count == self.reprod_mini_batch_size*self.reprod_batch_size - 1:
            self.update_reproduction(i_evolution)
            self.replay_buffer_reproducer.reset_buffer()
            return 'Agent_id:{}, Update status:{}'.format(self.agent_id,True)
        else:
            return 'Agent_id:{}, Update status:{}'.format(self.agent_id,False)


    def update_reproduction(self, total_steps, replay_buffer = 0):

        if replay_buffer == 0:
            replay_buffer = self.replay_buffer_reproducer
        
        batch = replay_buffer.get_training_data()
        
        approx_kl_divs = []
        self.total_reprod_loss = 0
        self.total_reprod_entropy_loss = 0
        
        # Optimize policy for K epochs:
        if self.use_shuffle:
            shuffle_times = self.K_epochs
            sampler = SubsetRandomSampler(range(self.reprod_batch_size))
        else:
            shuffle_times = self.K_epochs
            sampler = SequentialSampler(range(self.reprod_batch_size))
        for _ in range(shuffle_times):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(sampler, self.reprod_mini_batch_size, False):
                # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                
                self.reproducer.reset_rnn_hidden()
                
                
                now = self.reproducer.forward(batch['s'][index]) 
                dist_now = Categorical(logits=now)
                

                dist_entropy = dist_now.entropy() # shape(mini_batch_size, max_episode_len)
                a_logprob_now = dist_now.log_prob(batch['a'][index])  # shape(mini_batch_size, max_episode_len)
                ratios = torch.exp(a_logprob_now - batch['a_logprob'][index])
                # a/b=exp(log(a)-log(b))
                 # shape(mini_batch_size, max_episode_len)
                print(ratios.shape)
                # actor loss
                surr1 = ratios * batch['r'][index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch['r'][index]
                reproducer_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size, max_episode_len)
                reproducer_loss = (reproducer_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = a_logprob_now - batch['a_logprob'][index]
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).numpy()
                    approx_kl_divs.append(approx_kl_div)

                
                
                # Update
                self.optimizer_reproducer.zero_grad()
                
                reproducer_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.reproducer.parameters(), 0.5)
                self.optimizer_reproducer.step()
                
                
                self.total_reprod_loss += reproducer_loss 
                self.total_reprod_entropy_loss += dist_entropy.mean()
            
            self.reproducer.reset_rnn_hidden()
            print('REPRODUCTION LOSS: ',reproducer_loss)

    def policy_mapping(self):
        if self.policy_dist == "Discrete":
            return 0
        elif self.policy_dist == "Discrete RNN":
            return 1
        elif self.policy_dist == "Gaussian":
            return 2

    def policy_decoding(self,action):
        if action == 0:
            return "Discrete"
        elif action == 1:
            return "Discrete RNN"
        elif action == 2:
            return "Gaussian"


    def add_to_buffer(self, input_):
        self.replay_buffer.replace(input_)

    def get_buffer(self):
        return self.replay_buffer
    
    def explained_variance(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        '''
        Computes fraction of variance that ypred explains about y.
        Returns 1 - Var[y-ypred] / Var[y]

        interpretation:
            ev=0  =>  might as well have predicted zero
            ev=1  =>  perfect prediction
            ev<0  =>  worse than just predicting zero

        :param y_pred: the prediction
        :param y_true: the expected value
        :return: explained variance of ypred and y
        '''
        assert y_true.ndim == 1 and y_pred.ndim == 1
        var_y = np.var(y_true)
        return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


    def get_losses(self):
        if type(self.total_actor_loss) != int:
            return {'actor_loss':self.total_actor_loss.item(), 
                    'critic_loss':self.total_critic_loss.item(),
                    'entropy_loss':self.total_entropy_loss.item()}
        else:
            return {'actor_loss':self.total_actor_loss, 
                    'critic_loss':self.total_critic_loss,
                    'entropy_loss':self.total_entropy_loss
                    }
    
    def clone(self, id_):
        clone_ = copy.deepcopy(self)
        clone_.agent_id = id_
        return clone_

    def is_early_stopped(self):
        return self.stop_training

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_encoder.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
            