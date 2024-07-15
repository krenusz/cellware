import numpy as np
import torch
import torch.nn.functional as F

from state_action_reward_emb import *
from policies import *
from replay_buffer import ReplayBuffer, k_catche

class Encoder():
    def __init__(self, args, use_cross_attention=True):
        self.replay_buffer = copy.copy(ReplayBuffer(args))
        self.k_catche = k_catche.remote()
        self.device = torch.device("cuda" if torch.cuda.device_count()>0 and args.use_cuda else "cpu")
        self.dtype = torch.float32 if self.device.type == 'cuda' else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.device_count()>0 and args.use_cuda and args.use_mixprecision)
        self.encoder_model = RNN_Encoder(args).to(self.device)
        self.target_encoder_model = copy.deepcopy(self.encoder_model)
        self.optimizer = torch.optim.Adam(self.encoder_model.parameters(), lr=args.lr_e)
        self.softmax = nn.Softmax(dim=-1)
        self.reg_coeff = args.reg_coeff
        self.use_cross_attention = use_cross_attention
        self.args = args
        self.update_count = 0
        self.i_m = 0
        self.stop_training = False
        self.mean_kl_divergence = []

        
    def train(self, replay_buffer=None):
        for p in self.target_encoder_model.parameters():
            p.requires_grad = False

        if replay_buffer is None:
            replay_buffer = self.replay_buffer
        
        batch = replay_buffer.get_training_data()

        batch = {k: v.to(device=self.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}
        
        k_cross = ray.get(self.k_catche.get.remote())
        k_cross_batch = None
        
        if k_cross is not None:
            k_cross = k_cross.to(self.device)
            k_cross = k_cross.repeat(16, 1, 64, 1)
            
        loss_jepa, loss_reg, total_encoder_loss = 0., 0., 0.
        approx_kl_divs = []
        # Optimize policy for K epochs:
        if self.args.use_shuffle:
            shuffle_times = self.args.K_epochs_enc
            sampler = SubsetRandomSampler(range(self.args.batch_size))
            self.iter_batch = self.args.batch_size//4
        else:
            shuffle_times = self.args.K_epochs_enc
            sampler = SequentialSampler(range(self.args.batch_size))
            self.iter_batch = self.args.batch_size//4
        for i in range(shuffle_times):
            
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(sampler, self.iter_batch, False):
                if k_cross is not None:
                    k_cross_batch =  k_cross[index]
                s, a, r = batch['s'][index].cpu().numpy(), batch['a'][index].cpu().numpy(), batch['r'][index].cpu().numpy()
                s_, a_, r_ = np.zeros_like(s), np.zeros_like(a), np.zeros_like(r)
                s_[:,1:], a_[:,1:], r_[:,1:] = s[:,:-1], a[:,:-1], r[:,:-1]
                #print('shapes', s.shape, a.shape, r.shape, s_.shape, a_.shape, r_.shape)
                embedding_target = state_action_reward_embedding(s_, a_, r_, s, a, r, self.args.env_size, self.args.embed_dim, self.args.action_dim, self.args.max_reward)
                embedding_target = torch.tensor(embedding_target,dtype=self.dtype).to(self.device)
                embedding_masked = self.apply_mask(embedding_target)
                #print('state',s)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    
                   
                    if self.use_cross_attention:
                        with torch.no_grad():
                            x_target, _, _, _ = self.target_encoder_model.forward(embedding_target, k_cross_batch)
                        x, _, _, _ = self.encoder_model.forward(embedding_masked, k_cross_batch)
                    else:
                        with torch.no_grad():
                            x_target, _, _, _ = self.target_encoder_model.forward(embedding_target)
                        x, _, _, _ = self.encoder_model.forward(embedding_masked)
                
                    loss_jepa = self.loss_fn(x, x_target)
                    pstd_x = self.reg_fn(x)  # predictor variance across patches
                    loss_reg = torch.mean(F.relu(1.-pstd_x))
                    encoder_loss = loss_jepa + self.reg_coeff * loss_reg
                    encoder_loss = (encoder_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                    total_encoder_loss += encoder_loss.item()
                                          
                self.scaler.scale(encoder_loss).backward()
                if self.args.use_grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.encoder_model.parameters(), 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                ### I am happy to take suggestions to calculate KL divergence (early stopping) of two representation
                with torch.no_grad():
                    p = self.softmax(x_target)
                    q = self.softmax(x)
                    ratio = p / q
                    approx_kl_div = torch.sum(p * torch.log(ratio)).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

            # momentum update of target encoder
            # -- momentum schedule
            m = 0.998 + self.i_m*(1.-0.998)/(shuffle_times*self.args.max_train_steps/self.args.batch_size)
            with torch.no_grad():
                for param_q, param_k in zip(self.encoder_model.parameters(), self.target_encoder_model.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
            self.i_m += 1
            
        print('Encoder Loss:', total_encoder_loss/shuffle_times)
        print('Mean KL Divergence:', np.mean(approx_kl_divs))
        self.mean_kl_divergence.append(np.mean(approx_kl_divs))
        if self.update_count > 20:
            self.stop_training = self.early_stopping_criterion(approx_kl_divs)
        if self.stop_training:
            print(f"Early stopping at update count {self.update_count} due to reaching min kl differences")
        
        self.update_count += 1
    @torch.no_grad()
    def predict(self,x):
        return self.encoder_model.forward(x)

    def loss_fn(self, z, h, loss_exp=1.0):
        loss = 0.
        # Compute loss and accumulate for each mask-enc/mask-pred pair
        for zi, hi in zip(z, h):
            loss += torch.mean(torch.abs(zi - hi)**loss_exp, dim=-1) / loss_exp
        #loss /= len(masks_pred)
        return loss

    def reg_fn(self, z):
        return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)    
                
    def add_to_buffer(self, input_):
        self.replay_buffer.replace(input_)
    
    def apply_mask(self, embedding):
        
        embedding[:,:,-2*self.args.embed_dim:] = 0
        
        return embedding
    
    def early_stopping_criterion(self, kl_divergences):
        relative_change = abs(kl_divergences[-1] - kl_divergences[-2])
        print('KL Divergence differences',relative_change, '/ Threshold:', self.args.target_kl)
        return relative_change < self.args.target_kl