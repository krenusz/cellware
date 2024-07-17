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
import time
# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=0.7):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
def orthogonal_init_RNN(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

    return layer

class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.projection_embed_dim, args.hidden_width)
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
        self.fc1 = nn.Linear(args.projection_embed_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim_gauss)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim_gauss))  # We use 'nn.Parameter' to train log_std automatically
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
    
    def reinitilize(self, long_weights):
        self.mean_layer.weight = long_weights[0]
        self.fc1.weight = long_weights[1]
        self.fc2.weight = long_weights[2]

class Actor_Discrete(nn.Module):
    def __init__(self, args):
        super(Actor_Discrete, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.projection_embed_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        self.softmax = nn.Softmax(dim=-1)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # i would like distribution between 0,1 instead of -1,1
        a_probs = self.softmax(self.mean_layer(s)) 
        return a_probs

    def get_dist(self, s):
        a_probs = self.forward(s)
        dist = Categorical(a_probs)

        return dist
    
    def reinitilize(self, long_weights):
        self.mean_layer.weight = long_weights[0]
        self.fc1.weight = long_weights[1]
        self.fc2.weight = long_weights[2]



class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.projection_embed_dim, args.hidden_width)
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
        #print('s',s)
        v_s = self.fc3(s)
        return v_s

    def reinitilize(self, long_weights):
        
        self.fc1.weight = long_weights[0]
        self.fc2.weight = long_weights[0]
        self.fc3.weight = long_weights[0]


class Actor_RNN(nn.Module):
    def __init__(self, args):
        super(Actor_RNN, self).__init__()
        self.use_gru = args.use_gru
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        self.actor_rnn_hidden = None
        self.actor_fc1 = nn.Linear(args.projection_embed_dim, args.hidden_dim)
        if args.use_gru:
            print("------use GRU------")
            self.actor_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, num_layers = args.n_layers, batch_first=True)
        else:
            print("------use LSTM------")
            self.actor_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, num_layers = args.n_layers, batch_first=True)
        self.actor_fc2 = nn.Linear(args.hidden_dim, args.action_dim)
        self.softmax = nn.Softmax(dim=-1)
        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init_RNN(self.actor_fc1)
            orthogonal_init_RNN(self.actor_rnn)
            orthogonal_init_RNN(self.actor_fc2, gain=0.01)
   

    def forward(self, s, h = None):
        s = self.activate_func(self.actor_fc1(s))
        
        
        output, self.actor_rnn_hidden = self.actor_rnn(s, self.actor_rnn_hidden if h is None else h)
        
        
        logit = self.actor_fc2(output)
        
        return logit
    
    def get_dist(self, s, h = None):
        logit = self.forward(s, h)
        dist = Categorical(logits=logit)
        a = dist.sample()
        a_logprob = dist.log_prob(a)
        return a, a_logprob, logit
    def reset_rnn_hidden(self):
        self.actor_rnn_hidden = None

class RNN_reprod(nn.Module):
    def __init__(self, args):
        super(RNN_reprod, self).__init__()
        
        self.use_gru = args.use_gru
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        self.reprod_rnn_hidden = None
        self.reprod_fc1 = nn.Linear(args.state_dim_reprod, args.hidden_dim)
        if args.use_gru:
            print("------use GRU------")
            self.reprod_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, num_layers = args.n_layers, batch_first=True)
        else:
            print("------use LSTM------")
            self.reprod_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, num_layers = args.n_layers, batch_first=True)
        self.reprod_fc2 = nn.Linear(args.hidden_dim, args.action_dim_reprod)
        self.softmax = nn.Softmax(dim=-1)
        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init_RNN(self.reprod_fc1)
            orthogonal_init_RNN(self.reprod_rnn)
            orthogonal_init_RNN(self.reprod_fc2, gain=0.01)
   

    def forward(self, s, h = None):
        print('1',s.shape)
        s = self.activate_func(self.reprod_fc1(s))
        print('2',s.shape)
        
        output, self.reprod_rnn_hidden = self.reprod_rnn(s, self.reprod_rnn_hidden if h is None else h)
        
        
        logit = self.reprod_fc2(output)
        
        return logit
    
    def get_dist(self, s, h = None):
        logit = self.forward(s, h)
        dist = Categorical(logits=logit)
        a = dist.sample()
        a_logprob = dist.log_prob(a)
        return a, a_logprob, logit
    def reset_rnn_hidden(self):
        self.reprod_rnn_hidden = None
        


class Critic_RNN(nn.Module):
    def __init__(self, args):
        super(Critic_RNN, self).__init__()
        self.use_gru = args.use_gru
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        self.critic_rnn_hidden = None
        self.critic_fc1 = nn.Linear(args.projection_embed_dim, args.hidden_dim)
        if args.use_gru:
            self.critic_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
        else:
            self.critic_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)
        self.critic_fc2 = nn.Linear(args.hidden_dim, 1)

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init_RNN(self.critic_fc1)
            orthogonal_init_RNN(self.critic_rnn)
            orthogonal_init_RNN(self.critic_fc2)


    def forward(self, s, h = None):
        s = self.activate_func(self.critic_fc1(s))
        output, self.critic_rnn_hidden = self.critic_rnn(s, self.critic_rnn_hidden)
        #print('output',output)
        value = self.critic_fc2(output)
        return value
    def reset_rnn_hidden(self):
        self.critic_rnn_hidden = None

class RNN_Encoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, args):
        super(RNN_Encoder, self).__init__()  
        self.lin = nn.Linear(args.sar_embed_dim, args.rnn_encoder_embed_dim)
        if args.use_gru:
            self.rnn = nn.GRU(args.sar_embed_dim, args.rnn_encoder_embed_dim, args.n_layers, dropout=args.attn_drop, batch_first=False)
        else:
            self.rnn = nn.LSTM(args.sar_embed_dim, args.rnn_encoder_embed_dim, args.n_layers, dropout=args.attn_drop, batch_first=False)
        
        self.block = Block(args)
        self.args = args
        self.encoder_rnn_hidden = None

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init_RNN(self.rnn)
            orthogonal_init_RNN(self.lin)

    def forward(self, x, k_cross=None, decoder=True):
        
        B, MB, C = x.shape
        
        if self.args.use_rnn_encoder:
            dc_out, self.encoder_rnn_hidden = self.rnn(x, self.encoder_rnn_hidden)
        else:
            dc_out = self.lin(x)
        
        dc_out = dc_out.view(B, MB, -1)
               
        x, attn, k = self.block.forward(dc_out, k_cross, decoder)
        return x, attn, k, self.encoder_rnn_hidden
        
    def reset_rnn_hidden(self):
        self.encoder_rnn_hidden = None   

    def reinitilize(self, long_weights):
        #self.rnn.all_weights = long_weights[0]
        self.fc1.weight = long_weights[0]
        self.fc2.weight = long_weights[1]



class MLP(nn.Module):
    def __init__(self, args,
                 mlp_input_dim,mlp_output_size):
        super(MLP, self).__init__()
 

        self.add_module("input", nn.Linear(mlp_input_dim, args.hidden_dim))
        self.add_module(f"Use_Tanh{args.use_tanh}", nn.ReLU() if not args.use_tanh else nn.Tanh())
        for i in range(args.mlp_hidden_depth):
            self.add_module("hidden_"+str(i), nn.Linear(args.hidden_dim, args.hidden_dim))
            self.add_module(f"Use_Tanh{args.use_tanh}"+str(i), nn.ReLU() if not args.use_tanh else nn.Tanh())
            self.add_module("dropout_"+str(i), nn.Dropout(args.dropout_rate))
        self.add_module("output", nn.Linear(args.hidden_dim, mlp_output_size))
    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class Attention(nn.Sequential):
    def __init__(
        self, args,
        qkv_bias=False):

        super(Attention, self).__init__()
        self.num_heads = args.num_heads
        head_dim = args.predictor_embed_dim // self.num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(args.predictor_embed_dim, args.predictor_embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(args.attn_drop)
        self.proj = nn.Linear(args.predictor_embed_dim, args.predictor_embed_dim)
        self.proj_drop_prob = args.proj_drop
        self.proj_drop = nn.Dropout(args.proj_drop)
        self.use_sdpa = args.use_sdpa

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init_RNN(self.qkv)
            orthogonal_init_RNN(self.proj)

    def forward(self, x, mask=None):
        B, MB, C = x.shape
        
        qkv = self.qkv(x).reshape(B, MB, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, MB, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn, k
    



class CrossAttention(nn.Module): # ????????
    def __init__(
        self, args,        
        qkv_bias=False):

        super(CrossAttention, self).__init__()
        self.num_heads = args.num_heads
        head_dim = args.predictor_embed_dim // self.num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(args.predictor_embed_dim, args.predictor_embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(args.attn_drop)
        self.proj = nn.Linear(args.predictor_embed_dim, args.predictor_embed_dim)
        self.proj_drop_prob = args.proj_drop
        self.proj_drop = nn.Dropout(args.proj_drop)
        self.use_sdpa = args.use_sdpa

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init_RNN(self.qkv)
            orthogonal_init_RNN(self.proj)

    def forward(self, x, k_cross=None, mask=None):
        B, MB, C = x.shape
        
        qkv = self.qkv(x).reshape(B, MB, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if k_cross is None:
            k_cross = k
        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                
                x = F.scaled_dot_product_attention(q, k_cross, v, dropout_p=self.proj_drop_prob)
                attn = None
        else:
            attn = (q @ k_cross.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, MB, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn, k

class Block(nn.Sequential):
    def __init__(self, args, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()

        self.att_hidden_depth = args.att_hidden_depth
        self.predictor_embed = nn.Linear(args.rnn_encoder_embed_dim, args.predictor_embed_dim)
        self.norm1 = norm_layer(args.predictor_embed_dim)
        self.attantion = Attention(args)
        self.cross_attantion = CrossAttention(args)
        self.norm2 = norm_layer(args.predictor_embed_dim)  
        self.mlp = MLP(args, args.predictor_embed_dim, args.projection_embed_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init_RNN(self.predictor_embed)
            

    def forward(self, x, k_cross=None, decoder=True):
        x = self.activate_func(self.predictor_embed(x))
        
        for _ in range(self.att_hidden_depth):
            if k_cross is None:
                y, attn, k = self.attantion.forward(self.norm1(x))
            else:
                y, attn, k = self.cross_attantion.forward(self.norm1(x), k_cross)
            x = x + self.activate_func(y)
        
        
        z = self.norm2(x)
        z = self.mlp(z)
        
        return z, attn, k
        

