from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
import torch.nn as nn
from torch.distributions import Beta, Normal, Categorical
from replay_buffer import ReplayBuffer
import copy
import ray

from policies import Actor_Discrete, Actor_RNN, Actor_Beta, Actor_Gaussian, Critic, Critic_RNN, RNN_Encoder

@ray.remote
class PPO_universal():
    def __init__(self, args, env, agent_id):
        self.agent_id = agent_id
        self.env = env
        self.replay_buffer = copy.copy(ReplayBuffer(args))
        #self.replay_buffer = ReplayBuffer(args)  # Create replay buffer
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.max_episode_steps = args.max_episode_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
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

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5, maximize=False)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
            self.optimizer_encoder = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
            self.optimizer_encoder = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

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
                dist, weights = self.actor.get_dist(final)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
        elif self.policy_dist == "Discrete RNN":
            with torch.no_grad():
                a, a_logprob, logit = self.actor.get_dist(final)
                if evaluate:
                    a = torch.argmax(logit)
                    return a.item(), None
                
                    
        elif self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(final)
                
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist, weights = self.actor.get_dist(final)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, 0, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
                
        return a.numpy().flatten(), a_logprob.numpy().flatten(), h_end.numpy().flatten()
    def update_rnn(self, total_steps, replay_buffer = 0):
        if replay_buffer == 0:
            replay_buffer = self.replay_buffer
        
        batch = replay_buffer.get_training_data()


        #self.total_advantage = sum(adv).item()
        self.total_actor_loss = 0
        self.total_critic_loss = 0
        # Optimize policy for K epochs:
        if self.use_shuffle:
            shuffle_times = self.K_epochs
            sampler = SubsetRandomSampler(range(self.batch_size))
        else:
            shuffle_times = self.K_epochs
            sampler = SequentialSampler(range(self.batch_size))
        for _ in range(shuffle_times):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(sampler, self.collective_switch, False):
                #print(index)
                #if 0 in index:
                #    av_resource = torch.cat((torch.tensor([[1]]),res[[i-1 for i in index if i > 0]]))
                #else:
                #    av_resource = res[np.array(index)-1]

                # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                self.reset_rnn_hidden()
                #print(batch['s'][index].shape,batch['h'][index].shape)
                logits_now = self.actor.forward(torch.cat((batch['s'][index],batch['h'][index].view(self.batch_size,self.mini_batch_size,1)),dim=-1))  # logits_now.shape=(mini_batch_size, max_episode_len, action_dim)
                values_now = self.critic.forward(torch.cat((batch['s'][index],batch['h'][index].view(self.batch_size,self.mini_batch_size,1)),dim=-1)).squeeze(-1)  # values_now.shape=(mini_batch_size, max_episode_len)

                dist_now = Categorical(logits=logits_now)
                
                dist_entropy = dist_now.entropy()  # shape(mini_batch_size, max_episode_len)
                a_logprob_now = dist_now.log_prob(batch['a'][index])  # shape(mini_batch_size, max_episode_len)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - batch['a_logprob'][index])  # shape(mini_batch_size, max_episode_len)

                # actor loss
                surr1 = ratios * batch['adv'][index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch['adv'][index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size, max_episode_len)
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                # critic_loss
                critic_loss = (values_now - batch['r'][index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                
                
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
                print('LOSS/K_epoch:',loss, 'Total Reward:', batch['r'][index].sum(), 'Total Resource:', batch['res'][index].sum(), 'Total Adv:', batch['adv'][index].sum(), 'Total Pred Reward:', batch['v_target'][index].sum())
        self.reset_rnn_hidden()
        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)


    def update(self, total_steps, replay_buffer = 0):
        if replay_buffer == 0:
            replay_buffer = self.replay_buffer
        short_s, short_h, short_a, short_a_logprob, short_r, short_res, short_c, short_s_= replay_buffer.numpy_to_tensor()  # Get training data
        long_s, long_h, long_a, long_a_logprob, long_r, long_res, long_c, long_s_, long_weights = replay_buffer.longterm_to_tensor()
        print('Long Term Memory sum reward', long_r.sum() , 'Short Term Memory sum reward', short_r.sum())

        if total_steps >= 10000:
            # Exploitation Phase
            self.use_lr_decay = True
            sample_size = int(np.round(self.batch_size - self.batch_size*total_steps/self.max_train_steps))
            long_index = np.random.choice(self.batch_size, size=self.batch_size-sample_size, replace=False)
            short_index = np.random.choice(self.batch_size, size=sample_size, replace=False)
            print('long_index', len(long_index), 'short_index', len(short_index))
            s = torch.cat((short_s[short_index],long_s[long_index]),dim=0)
            h = torch.cat((short_h[short_index],long_h[long_index]),dim=0)
            a = torch.cat((short_a[short_index],long_a[long_index]),dim=0)
            a_logprob = torch.cat((short_a_logprob[short_index],long_a_logprob[long_index]),dim=0)
            r = torch.cat((short_r[short_index],long_r[long_index]),dim=0)
            res = torch.cat((short_res[short_index],long_res[long_index]),dim=0)
            s_ = torch.cat((short_s_[short_index],long_s_[long_index]),dim=0)
        else:
            # Exploration Phase
            s = short_s
            h = short_h
            a = short_a
            a_logprob = short_a_logprob
            r = short_r
            res = short_res
            s_ = short_s_

        #replay_buffer.save_memory('runs/PPO_continuous/env_{}_level_{}_dist_{}_numberofagent_{}_collective_{}/{}_{}'.format(self.env_name, self.lvl, self.policy_dist, self.agent_number, self.use_collective, total_steps, self.agent_id))
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """



        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(torch.cat((s,h),dim=1))
            vs_ = self.critic(torch.cat((s_,h),dim=1))
            deltas = r + self.gamma * vs_ - vs
            for i, delta in enumerate(reversed(deltas.flatten().numpy())):
                
                gae = delta #+ self.gamma * self.lamda * gae
                
                #if i % (self.max_episode_steps - 1) == 0:
                #    gae = 0
                
                
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
                res = ((res - adv.mean()) / (res.std() + 1e-5))
        self.total_advantage = sum(adv).item()
        self.total_actor_loss = 0
        self.total_critic_loss = 0
        # Optimize policy for K epochs:
        if self.use_shuffle:
            shuffle_times = self.K_epochs
            sampler = SubsetRandomSampler(range(self.batch_size))
        else:
            shuffle_times = 1
            sampler = SequentialSampler(range(self.batch_size))
        for _ in range(shuffle_times):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(sampler, self.collective_switch, False):
                
                if 0 in index:
                    av_resource = torch.cat((torch.tensor([[1]]),res[[i-1 for i in index if i > 0]]))
                else:
                    av_resource = res[np.array(index)-1]
                #print(av_resource)
                #av_resource = torch.where(av_resource < 0, -(av_resource ** 2), av_resource)
                #if self.use_adv_norm:  # normalization
                #    av_resource = ((av_resource - av_resource.mean()) / (av_resource.std() + 1e-5))

                dist_now, _ = self.actor.get_dist(torch.cat((s[index],h[index]),dim=1))
                if self.policy_dist == "Discrete":
                    dist_entropy = dist_now.entropy().sum(-1, keepdim=True)  # shape(mini_batch_size X 1)
                else:
                    dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])

                v_s = self.critic(torch.cat((s[index],h[index]),dim=1))
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.mean(1, keepdim=True) - a_logprob[index].mean(1, keepdim=True))  # shape(mini_batch_size X 1)

                #surr1 = ratios *  adv[index] - r[index] # Only calculate the gradient of 'a_logprob_now' in ratios
                #actor_loss = -surr1 #- self.entropy_coef * dist_entropy
                #surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index] - r[index]
                #actor_loss = -torch.min(surr1, surr2) + 0.5 * abs(F.mse_loss(r[index], v_s)) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                actor_loss = -r[index] * ratios + self.entropy_coef * dist_entropy
                # Update actor and encoder
                self.optimizer_actor.zero_grad()
                self.optimizer_encoder.zero_grad()
                self.optimizer_critic.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_actor.step()
                self.optimizer_encoder.step()
                self.optimizer_critic.step()
                self.total_actor_loss += actor_loss
                                                           
                

        print('Mean total_actor_loss', self.total_actor_loss.flatten().mean(), 'Mean Actor Ratios', ratios.mean(),'Mean Advantage', adv.mean(), 'total_advantage', self.total_advantage,'sum reward', r.sum(), 'sum resources', res.sum())  
        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)
        #if total_steps >= 100:
        #    self.actor.reinitilize(long_weights)
        
    def act(self, state, i_epoch, lvl):
        
        from threadpoolctl import threadpool_limits
    
        
        with threadpool_limits(limits=1, user_api='blas'):
            score = 0
            
            if self.done:
                print('done', 'agent', self.agent_id)
                return state, self.done
            
                      
                        
            a, action_prob, h_end = self.choose_action(state, i_epoch)

            value = self.get_value(torch.cat((torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0),torch.unsqueeze(torch.tensor(h_end, dtype=torch.float), 0)),dim=1))
            
            a = a.round()
            if self.policy_dist == "Beta":
                action = 2 * (a - 0.5) * max(self.env.actions)  # [0,1]->[-max,max]
            else:  
                action = a
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
                #print('done: agent getting updated')
                if self.policy_dist == "Discrete RNN":
                    self.update_rnn(i_epoch)
                    self.replay_buffer.reset_buffer()
                    print('buffer was reseted',self.replay_buffer.episode_num, self.replay_buffer.episode_step)
                    
                else:
                    self.update(i_epoch)
                    self.replay_buffer.reset_buffer()
        if lvl == 1:
            return next_state, self.score
        else:
            return next_state, self.done
    
    def add_to_buffer(self, input_):
        self.replay_buffer.replace(input_)

    def get_buffer(self):
        return self.replay_buffer

    def get_losses(self):
        if type(self.total_actor_loss) != int:
            return sum(self.total_actor_loss).item(), sum(self.total_actor_loss).item(), self.total_advantage
        else:
            return self.total_actor_loss, self.total_actor_loss, self.total_advantage
    
    def clone(self, id_):
        clone_ = copy.deepcopy(self)
        clone_.agent_id = id_
        return clone_

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_encoder.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
            