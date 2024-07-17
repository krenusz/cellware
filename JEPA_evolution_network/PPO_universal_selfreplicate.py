
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
import torch.nn as nn
from torch.distributions import Beta, Normal, Categorical
from replay_buffer import ReplayBuffer, k_catche
from replay_buffer_reproducer import ReplayBuffer_reproducer
from state_action_reward_emb import *
import copy
import ray
from threadpoolctl import threadpool_limits

from policies import *

@ray.remote(num_gpus=1)
class PPO_universal_selfreplicate():
    def __init__(self, args, env, agent_id, policy_dist, encoder):
        self.encoder = encoder
        self.agent_id = agent_id
        self.env = env
        self.replay_buffer = copy.copy(ReplayBuffer(args))
        self.k_catche = k_catche.remote()
        self.replay_buffer_reproducer = copy.copy(ReplayBuffer_reproducer(args))
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
        self.use_cross_attention = args.use_cross_attention
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
        self.args = args
        
        self.device = torch.device("cuda" if torch.cuda.device_count()>0 and args.use_cuda else "cpu")
        
        self.dtype = torch.float32 if self.device.type == 'cuda' else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.device_count()>0 and args.use_cuda and args.use_mixprecision)
        
        print('Cuda infos 1:',torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(), self.device, self.dtype)

        if self.policy_dist == "Discrete":
            self.actor = Actor_Discrete(args).to(self.device)
            self.critic = Critic(args).to(self.device)
        elif self.policy_dist == "Discrete RNN":
            self.actor = Actor_RNN(args).to(self.device)
            self.critic = Critic_RNN(args).to(self.device)
        elif self.policy_dist == "Beta":
            self.actor = Actor_Beta(args).to(self.device)
            self.critic = Critic(args).to(self.device)
        else:
            self.actor = Actor_Gaussian(args).to(self.device)
            self.critic = Critic(args).to(self.device)

        self.reproducer = RNN_reprod(args).to(self.device)
        
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5, maximize=False)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
            self.optimizer_reproducer = torch.optim.Adam(self.reproducer.parameters(), lr=self.lr_r, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
            self.optimizer_reproducer = torch.optim.Adam(self.reproducer.parameters(), lr=self.lr_r)

    def upgrade_encoder(self, encoder):
        self.encoder = encoder

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
    def apply_mask(self, embedding):
        
        embedding[:,:,-2*self.args.embed_dim:] = 0
        
        return embedding
    
    def get_value(self, s):
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):

                #s = torch.tensor(s, dtype=self.dtype, device=self.device).unsqueeze(0)
                value = self.critic(s)
            return value.cpu().item()
    
    @torch.no_grad()
    def choose_action(self, state, i_epoch, evaluate=False):
        episode_step = self.replay_buffer.episode_step
        episode_num = self.replay_buffer.episode_num
          
        state = np.asarray(state).reshape(1,1,-1) 

        batch = self.replay_buffer.numpy_()
       
        s,a,r = batch['s'][episode_num,episode_step-1], batch['a'][episode_num,episode_step-1], batch['r'][episode_num,episode_step-1]
        s,a,r = s.reshape(1,1,-1),a.reshape(1,-1),r.reshape(1,-1)

        k_cross = ray.get(self.k_catche.get.remote())
        if k_cross is not None:
            k_cross = k_cross.to(self.device)
            #print(k_cross.shape)
        embedding = state_action_reward_embedding(s, a, r, state, size=self.args.env_size, embed_dim=self.args.embed_dim, num_action=self.args.action_dim, max_reward=self.args.max_reward)
        embedding = torch.tensor(embedding,dtype=self.dtype).view(1,1,-1).to(self.device)
        
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            

            x, _, _, _ = self.encoder.predict(embedding)

            if self.policy_dist == "Discrete":
                dist = self.actor.get_dist(x)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
            elif self.policy_dist == "Discrete RNN":
                a, a_logprob, logit = self.actor.get_dist(x)
                if evaluate:
                    a = torch.argmax(logit)
                    return a.item(), None
            elif self.policy_dist == "Gaussian": 
                dist = self.actor.get_dist(x)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, 0, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)            
            elif self.policy_dist == "Beta":
                raise ValueError('Beta distribution not supported yet')
                dist = self.actor.get_dist(final)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
            else:
                raise ValueError('Policy distribution not recognized')
        
        return a.cpu().float().numpy().flatten(), a_logprob.cpu().float().numpy().flatten(), x

    def update(self, total_steps, replay_buffer = 0):
        
            
        if replay_buffer == 0:
            replay_buffer = self.replay_buffer
        
        batch = replay_buffer.get_training_data()

        batch = {k: v.to(device=self.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}
        print('Updating Agent with policy:', self.policy_dist, 'Agent:', self.agent_id)
        k_cross = ray.get(self.k_catche.get.remote())
        
        # get the best action fer all the states based on the reward for it
        max_indices = batch['r'].argmax(dim=0).to(self.device)
        col_indices = torch.arange(batch['r'].size(1)).unsqueeze(1).to(self.device)
        indices = torch.stack([max_indices.flatten(), col_indices.flatten()], dim=1)
        best_actions = batch['a'][indices[:, 0], indices[:, 1]].reshape(1, self.mini_batch_size).to(self.device)

        if k_cross is not None:
            k_cross = k_cross.to(self.device)
            
            k_cross = k_cross.repeat(16, 1, 64, 1)
          
        approx_kl_divs = []
        self.total_encoder_loss = 0
        self.total_actor_loss = 0
        self.total_critic_loss = 0
        self.total_advantage = 0
        self.total_entropy_loss = 0
        # Optimize policy for K epochs:
        if self.use_shuffle:
            shuffle_times = self.K_epochs
            sampler = SubsetRandomSampler(range(self.batch_size))
            self.iter_batch = self.batch_size//4
        else:
            shuffle_times = self.K_epochs
            sampler = SequentialSampler(range(self.batch_size))
            self.iter_batch = self.batch_size//4
        for _ in range(shuffle_times):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(sampler, self.iter_batch, False):
                s, a, r = batch['s'][index].cpu().numpy(), batch['a'][index].cpu().numpy(), batch['r'][index].cpu().numpy()
                s_, a_, r_ = np.zeros_like(s), np.zeros_like(a), np.zeros_like(r)
                s_[:,1:], a_[:,1:], r_[:,1:] = s[:,:-1], a[:,:-1], r[:,:-1]
                
                embedding_target = state_action_reward_embedding(s_, a_, r_, s, a, r, self.args.env_size, self.args.embed_dim, self.args.action_dim, self.args.max_reward)
                embedding_target = torch.tensor(embedding_target,dtype=self.dtype).to(self.device)
                embedding_masked = self.apply_mask(embedding_target)
                               
                if self.policy_dist == "Discrete RNN":
                    self.reset_rnn_hidden()
                if self.args.use_rnn_encoder:
                    self.encoder.encoder_model.reset_rnn_hidden()
                    

                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    with torch.no_grad():  

                        x, _, _, _= self.encoder.predict(embedding_masked)

                    if self.policy_dist == "Discrete RNN":
                        now = self.actor.forward(x)
                        dist_now = Categorical(logits=now)
                    elif self.policy_dist == "Gaussian":
                        dist_now = self.actor.get_dist(x)
                    elif self.policy_dist == "Discrete":
                        now = self.actor.forward(x)
                        dist_now = Categorical(probs=now)
                    else:
                        raise ValueError('Policy distribution not recognized (Beta distribution not supported yet)')
                    
                    values_now = self.critic.forward(x).squeeze(-1) 
                    if self.policy_dist == "Gaussian":
                        dist_entropy = dist_now.entropy().squeeze(dim=-1) # shape(mini_batch_size, max_episode_len)
                        a_logprob_now = dist_now.log_prob(batch['a'][index].view(self.batch_size,self.mini_batch_size,1)).squeeze(dim=-1)
                        ratios = torch.exp(a_logprob_now - batch['a_logprob'][index]) 
                    else:
                        dist_entropy = dist_now.entropy() # shape(mini_batch_size, max_episode_len)
                        #a_logprob_now = dist_now.log_prob(best_actions.repeat(self.iter_batch,1))
                        a_logprob_now = dist_now.log_prob(batch['a'][index])  # shape(mini_batch_size, max_episode_len)
                        ratios = torch.exp(a_logprob_now - batch['a_logprob'][index])
                    # a/b=exp(log(a)-log(b))
                    
                    # actor loss
                    surr1 = ratios * batch['adv'][index]
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch['adv'][index]
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size, max_episode_len)
                    actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                    
                    # critic_loss
                    critic_loss = (values_now - batch['r'][index]) ** 2
                    critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                    
                loss = actor_loss + critic_loss * 0.5

                self.scaler.scale(loss).backward()

                if self.use_grad_clip:
                    self.scaler.unscale_(self.optimizer_actor)
                    self.scaler.unscale_(self.optimizer_critic)
  
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.scaler.step(self.optimizer_actor)
                self.scaler.step(self.optimizer_critic)
                
                # Update
                self.scaler.update() 

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()

                self.total_actor_loss += actor_loss
                self.total_critic_loss += critic_loss
                self.total_entropy_loss += dist_entropy.mean()
                self.total_advantage += batch['adv']
       
        if self.policy_dist == "Discrete RNN":
            self.reset_rnn_hidden()
        if self.args.use_rnn_encoder:
            self.encoder.encoder_model.reset_rnn_hidden()
        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    
    def act(self, state, i_epoch, lvl):
           
        
        with threadpool_limits(limits=1, user_api='blas'):
            score = 0
            
            if self.done:
                print('done', 'agent', self.agent_id)
                return state, self.done
            
            #print('BS, epnum,count',self.batch_size,self.replay_buffer.episode_num, self.replay_buffer.count)
            if self.replay_buffer.episode_num == self.batch_size:
                #print('DEBUG')
                if self.use_collective:
                    self.update(i_epoch)
                    self.replay_buffer.reset_buffer()
                else:
                    #print('MEMORY RESETED')
                    self.update(i_epoch)
                    self.replay_buffer.reset_buffer()
                      
                        
            a, action_prob, x = self.choose_action(state, i_epoch)
            
            input_critic =  x.view(1,1,-1)
            value = self.get_value(input_critic)
            
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
            
            if self.replay_buffer.episode_step == self.mini_batch_size-1:
                dw = True
                self.replay_buffer.store_last_value(state, value, action, action_prob, reward, reward-cost, cost, dw)
                
            else:
                dw = False
                self.replay_buffer.store_transition(state, value, action, action_prob, reward, reward-cost, cost, dw)
        
        if lvl == 1:
            return next_state, self.score
        else:
            return next_state, self.done
    
    def select_differentiation_action(self):
        from threadpoolctl import threadpool_limits
    
        state = self.policy_mapping()
        with threadpool_limits(limits=1, user_api='blas'):
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    input_ = torch.tensor([state], dtype=self.dtype, device=self.device).unsqueeze(dim=0)
                    a, a_logprob, logit = self.reproducer.get_dist(input_)

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
                self.actor = Actor_Discrete(args).to(self.device)
                self.critic = Critic(args).to(self.device)
            elif self.policy_dist == "Discrete RNN":
                self.actor = Actor_RNN(args).to(self.device)
                self.critic = Critic_RNN(args).to(self.device)
            elif self.policy_dist == "Beta":
                self.actor = Actor_Beta(args).to(self.device)
                self.critic = Critic(args).to(self.device)
            else:
                self.actor = Actor_Gaussian(args).to(self.device)
                self.critic = Critic(args).to(self.device)
            
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
        batch = {k: v.to(device=self.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}
        
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
                
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    now = self.reproducer.forward(batch['s'][index]) 
                    dist_now = Categorical(logits=now)
                    

                    dist_entropy = dist_now.entropy() # shape(mini_batch_size, max_episode_len)
                    a_logprob_now = dist_now.log_prob(batch['a'][index])  # shape(mini_batch_size, max_episode_len)
                    ratios = torch.exp(a_logprob_now - batch['a_logprob'][index])
                    # a/b=exp(log(a)-log(b))
                    
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
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                
                
                self.optimizer_reproducer.zero_grad()
        
                self.scaler.scale(reproducer_loss).backward()
                
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    self.scaler.unscale_(self.optimizer_reproducer)
                    torch.nn.utils.clip_grad_norm_(self.reproducer.parameters(), 0.5)
                
                #self.optimizer_reproducer.step()
                self.scaler.step(self.optimizer_reproducer)
                # Update
                self.scaler.update()

                
                
                self.total_reprod_loss += reproducer_loss 
                self.total_reprod_entropy_loss += dist_entropy.mean()
            
            self.reproducer.reset_rnn_hidden()
            print('REPRODUCTION LOSS: ',reproducer_loss)

    def policy_mapping(self):
        if self.policy_dist == "Transformer":
            return 0
        elif self.policy_dist == "Discrete":
            return 1
        elif self.policy_dist == "Discrete RNN":
            return 2
        elif self.policy_dist == "Gaussian":
            return 3

    def policy_decoding(self,action):
        if action == 0:
            return "Transformer"
        elif action == 1:
            return "Discrete"
        elif action == 2:
            return "Discrete RNN"
        elif action == 3:
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
        #for p in self.optimizer_encoder.param_groups:
        #    p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
            