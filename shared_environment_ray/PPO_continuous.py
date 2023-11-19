from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
from replay_buffer import ReplayBuffer
import copy
import ray

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # i would like distribution between 0,1 instead of -1,1
        mean = self.max_action * torch.sigmoid(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s

@ray.remote
class PPO_continuous():
    def __init__(self, args, env, agent_id):
        self.agent_id = agent_id
        self.env = env
        self.replay_buffer = copy.copy(ReplayBuffer(args))
        #self.replay_buffer = ReplayBuffer(args)  # Create replay buffer
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
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
        
        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        else:
            self.actor = Actor_Gaussian(args)
        self.critic = Critic(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5, maximize=False)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def achievement_reset(self):
        self.done = False
        self.score = 0

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).detach().numpy().flatten()
        else:
            a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, 0, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, total_steps, replay_buffer = 0):
        if replay_buffer == 0:
            replay_buffer = self.replay_buffer
        s, a, a_logprob, r, s_= replay_buffer.numpy_to_tensor()  # Get training data
        #replay_buffer.save_memory('runs/PPO_continuous/env_{}_level_{}_dist_{}_numberofagent_{}_collective_{}/{}_{}'.format(self.env_name, self.lvl, self.policy_dist, self.agent_number, self.use_collective, total_steps, self.agent_id))
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """

        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * vs_ - vs
            for i, delta in enumerate(reversed(deltas.flatten().numpy())):
                
                gae = delta + self.gamma * self.lamda * gae
                
                if i % (self.max_episode_steps - 1) == 0:
                    gae = 0
                
                
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
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
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()
                self.total_actor_loss += actor_loss
                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
                self.total_critic_loss += critic_loss
        print('total_actor_loss', self.total_actor_loss, 'total_critic_loss', self.total_critic_loss, 'total_advantage', self.total_advantage)
        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)
        
    def act(self, state, i_epoch, lvl):
        
        from threadpoolctl import threadpool_limits
    
        
        with threadpool_limits(limits=1, user_api='blas'):
            score = 0
            
            if self.done:
                print('done', 'agent', self.agent_id)
                self.replay_buffer.reset() # clear experience
                return state, self.done
            
            if self.replay_buffer.count >= self.batch_size:
                #print('done: agent getting updated')
                self.update(i_epoch)
                self.replay_buffer.reset()

            a, action_prob = self.choose_action(state)
            a = a.round()
            if self.policy_dist == "Beta":
                action = 2 * (a - 0.5) * max(self.env.actions)  # [0,1]->[-max,max]
            else:  
                action = a

            next_state, reward_lvl1, reward_lvl2 = ray.get(self.env.step.remote(action,state))
            
            if lvl == 1:
                reward = reward_lvl1
            else:
                reward = reward_lvl2
            #print('reward',reward, 'lvl', lvl, 'cnt', cnt, 'i_epoch', i_epoch,'memory_state',self.replay_buffer.count)
            self.replay_buffer.store(state, action, action_prob, reward, next_state)
            #print('memory stored', self.replay_buffer.count)

            if reward == 90 and lvl == 1:
                self.score += 1
                print('CONDITION: food collected')
            elif reward == 90 and lvl == 2:
                self.done = True
                self.update(i_epoch)
                print('place taken by', self.agent_id)

            elif reward != 90 and lvl == 1:
                self.score = 0
                print('CONDITION: food not collected')
            elif reward != 90 and lvl == 2:
                self.done = False

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
            return sum(self.total_actor_loss).item(), self.total_critic_loss.item(), self.total_advantage
        else:
            return self.total_actor_loss, self.total_critic_loss, self.total_advantage
    
    def clone(self, id_):
        clone_ = copy.deepcopy(self)
        clone_.agent_id = id_
        return clone_

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
            