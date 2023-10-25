import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

import ray
import argparse
from collections import namedtuple
from itertools import count

from replay_buffer import ReplayBuffer
from PPO_continuous import PPO_continuous
from environment_dish import Environment
from cell2 import cells1

ray.init()
#ignore_reinit_error=True, address='auto', runtime_env={"working_dir": "C:/Users/User/.conda/envs/mtgat"}a
def main(args, env_name, number):
    
    env = Environment.remote(10,10)
    lvl = 2
    args.collective_switch = 512
    args.state_dim = len(ray.get(env.get_attributes.remote())[1])
    args.action_dim = 1
    args.max_action = 3
    args.max_episode_steps = ray.get(env.get_attributes.remote())[4]
    #args.writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_level_{}_dist_{}_numberofagent_{}_collective_{}'.format(env_name,lvl,args.policy_dist, args.agent_number, args.use_collective))
   
    agent_list = []
    state_list = []
    score_list = []
    for i in range(args.agent_number):
        agent_list.append(PPO_continuous.remote(args,env,i))
        state_list.append(0)
        score_list.append(0)
        max_id = i
    for i_epoch in range(args.max_train_steps):
        print('i_epoch:',i_epoch)
        for s in range(len(agent_list)):
            agent_list[s].achievement_reset.remote()
            state_list[s] = np.array(np.array(ray.get(env.reset.remote())))
            score_list[s] = 0
        done_list = [False for x in range(len(agent_list))]
        time.sleep(2)

        
        a = ray.get(env.get_attributes.remote())[2].copy() * 10
        b = ray.get(env.get_attributes.remote())[3].copy() * 10
        

        
        
        for cnt in count():

            if cnt == args.max_episode_steps: 
                print('done: No more Steps left')
                break
            
            print('Buffer size',[ray.get(x.get_attributes.remote()).count for x in agent_list], 'Batch size:',args.batch_size)
            #print('Buffer size',[[y[:5] for y in x.replay_buffer.numpy_to_tensor()] for x in agent_list], 'Batch size:',agent_list[0].batch_size)
            
            if args.use_collective and any([ray.get(x.get_attributes.remote()).count == args.collective_switch for x in agent_list]):
                array_list = [ray.get(x.get_buffer.remote()).numpy_() for x in agent_list]
                collective = [[x[i][0:args.collective_switch] for x in array_list] for i in range(len(array_list[0]))]
                collective_buffer = [np.stack(x).reshape(-1,x[0].shape[1]) for x in collective]
                print('collective_buffer',collective_buffer)
                for i in range(len(agent_list)):
                    agent_list[i].add_to_buffer.remote(collective_buffer)
            
                
            
            print('Buffer size',[ray.get(x.get_attributes.remote()).count for x in agent_list], 'Batch size:',args.batch_size)
            
            output = [agent_list[i].act.remote(state_list[i], i_epoch, lvl) for i in range(len(agent_list))]
            result = ray.get(output)
            
            for c in range(len(agent_list)):
                state_list[c] = result[c][0]
                if lvl == 1:
                    a[state_list[c][0], state_list[c][1]] -= 1
                else:
                    b[state_list[c][0], state_list[c][1]] -= 1
            
                if lvl == 1:
                    score_list[c] += result[c][1]
                    print('mod env variables and scores ','cell:',c,'SCORE',score_list[c])
                else:
                    done_list[c] = result[c][1]
                    print('mod env variables and scores ','cell:',c,'DONE',done_list[c])
            
            #for i in range(len(agent_list)):
            #    if done_list[i] == True:
            #        #print('agent',i,'done')
            #        for j in range(len(agent_list)):
            #            
            #            if i != j:
            #                #print('sharing memory with agent',j)
            #                agent_list[j].buffer.extend(agent_list[i].buffer)
                            
            #print('next_state_list',state_list)
            if lvl == 1:
                print('REWARD MAP LVL1: \n', a)
                best_score_index = score_list.index(max(score_list)) 
            else:
                print('REWARD MAP LVL2: \n', b)


            

            if lvl == 1 and max(score_list) == 2 and len(agent_list) < 4:
                print('All food collected! next agent joining...','collecter: Agent',best_score_index)
                
                agent_list.append(ray.get(agent_list[best_score_index].clone.remote(max_id+1)))
                state_list.append(state_list[best_score_index])
                score_list.append(0)
                break

            
            
            if all(done_list) and lvl == 2:
                print('LEVEL 2 ACHIEVED : All places taken by separate Agent -- juhhuuuuu')
                raise ValueError('LEVEL 2 ACHIEVED : All places taken by separate Agent -- juhhuuuuu.')
                
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--agent_number", type=int, default=4, help="Number of agents")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_collective", type=bool, default=True, help="Sharing memory across agents")
    args = parser.parse_args()

    env_name = ['Environment Petri Dish']
    env_index = 1
    main(args, env_name=env_name[0], number=1)