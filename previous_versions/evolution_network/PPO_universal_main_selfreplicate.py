import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import time

import os
import ray
import argparse
from collections import namedtuple
from itertools import count


from PPO_universal_selfreplicate import PPO_universal_selfreplicate
from environment_dish import Environment


ray.init()
def main(args, env_name, number):

    env = Environment.remote(10,10)
    args.env_name = env_name
    args.lvl = 1
    
    args.state_dim = 4
    
    args.action_dim_gauss = 1
    
    args.action_dim = 5
    args.max_action = 4
    args.state_dim_reprod = 1
    args.action_dim_reprod = 3
    args.max_episode_steps = args.mini_batch_size
    writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_level_{}_dist_{}_numberofagent_{}_collective_{}_number_{}'.format(env_name,args.lvl,args.policy_dist, args.agent_number, args.use_collective, number))
    starting_states = [np.array([0,1,0,0])]
    
    agent_list = []
    state_list = []
    score_list = []
    evolution_reward = []
    evolution_loss = []
    log_count = 0
    for i in range(args.agent_number):
        agent_list.append(PPO_universal_selfreplicate.remote(args,env,'0_gen_{}'.format(i),args.policy_dist))
        state_list.append(0)
        score_list.append(0)
        evolution_reward.append(0)
        evolution_loss.append(0)
        max_id = i

    for i_evolution in range(args.evolution_lenght):
        
        
        output_reproduction = [agent_list[i].select_differentiation_action.remote() for i in range(len(agent_list))]
        result_reproduction = ray.get(output_reproduction)
        print('i_evolution:',i_evolution, 'Policies',result_reproduction)
        old_agent_list = agent_list
        
        for i in range(len(old_agent_list)):
            agent_list[i].differentiate.remote(args,result_reproduction[i][1],'{}_gen_{}'.format(i_evolution,i))
            evolution_reward[i] = 0
                            
        for i_epoch in range(args.max_train_steps):
            env.reset.remote()
            print('i_epoch:',i_epoch)
            for s in range(len(agent_list)):
                   
                agent_list[s].achievement_reset.remote()
                state_list[s] = starting_states[0]
                score_list[s] = 0
            done_list = [False for x in range(len(agent_list))]
            #time.sleep(2)

            
            a = ray.get(env.get_attributes.remote())[0].copy() * 10
            b = ray.get(env.get_attributes.remote())[1].copy() * 10
            
            if all([ray.get(x.is_early_stopped.remote()) for x in agent_list]) == True:
                raise ValueError('Early Stopping')
            
            
            for cnt in count():
                
                
                #print('Buffer size',[ray.get(x.get_buffer.remote()).count for x in agent_list], 'Batch size:',args.batch_size)
                if cnt == args.max_episode_steps: 
                    print('done: No more Steps left')
                    
                    array_list = [ray.get(x.get_buffer.remote()).numpy_() for x in agent_list]
                    #print('array_list',array_list[0])
                    route_list = [array_list[x]['s'][ray.get(agent_list[x].get_buffer.remote()).episode_num-1] for x in range(len(agent_list))]
                    action_list = [array_list[x]['a'][ray.get(agent_list[x].get_buffer.remote()).episode_num-1] for x in range(len(agent_list))]
                    logprob_list = [array_list[x]['a_logprob'][ray.get(agent_list[x].get_buffer.remote()).episode_num-1] for x in range(len(agent_list))]
                    reward_list = [array_list[x]['r'][ray.get(agent_list[x].get_buffer.remote()).episode_num-1] for x in range(len(agent_list))]
                    try:
                        os.mkdir('routes/number_{}_collective_{}'.format(number,args.use_collective))
                        np.save('routes/number_{}_collective_{}/episode{}_route_list.npy'.format(number,args.use_collective,log_count),route_list)
                    except:
                        np.save('routes/number_{}_collective_{}/episode{}_route_list.npy'.format(number,args.use_collective,log_count),route_list)
                    if not any(len(route_list[0]) != len(i) for i in route_list):
                        std_route = np.mean(np.mean(np.std(route_list,axis=0),axis=1))
                        writer.add_scalar('Standard deviation of Routes', std_route, log_count)
                    if not any(len(action_list[0]) != len(i) for i in action_list):
                        std_action = np.std(action_list)
                        epoch_cv = np.mean(np.std(logprob_list,axis=0)/np.mean(logprob_list,axis=0))
                        writer.add_scalar('Standard deviation of Actions', std_action, log_count)
                        writer.add_scalar('Coefficient of variation of Log probs', epoch_cv, log_count)
                    if not any(len(logprob_list[0]) != len(i) for i in logprob_list):
                        std_logprob = np.std(logprob_list)
                        writer.add_scalar('Standard deviation of Log probs', std_logprob, log_count)
                    if not any(len(reward_list[0]) != len(i) for i in reward_list):
                        std_reward = np.std(reward_list)
                        cv_total_reward = np.std(np.sum(reward_list,axis=1))/np.mean(np.sum(reward_list,axis=1))
                        writer.add_scalar('Standard deviation of Rewards', std_reward, log_count)
                        writer.add_scalar('Coefficient of variation of Total Episode Rewards', cv_total_reward, log_count)
                    
                    for i in range(len(agent_list)):
                        #print(ray.get(agent_list[i].get_losses.remote())[0],ray.get(agent_list[i].get_losses.remote())[1],ray.get(agent_list[i].get_losses.remote())[2])
                        writer.add_scalar('Total actor losses agents {}'.format(i),ray.get(agent_list[i].get_losses.remote())['actor_loss'] ,log_count)
                        writer.add_scalar('Total critic losses agents {}'.format(i),ray.get(agent_list[i].get_losses.remote())['critic_loss'] ,log_count)
                        #writer.add_scalar('Total advantages agents {}'.format(i),ray.get(agent_list[i].get_losses.remote())['advantage'] ,i_epoch)
                        writer.add_scalar('Total rewards agents {}'.format(i),np.sum(reward_list[i]),log_count)
                        writer.add_scalar('Total entropy agents {}'.format(i),ray.get(agent_list[i].get_losses.remote())['entropy_loss'] ,log_count)
                    break
                
                
                if args.use_collective and any([ray.get(x.get_buffer.remote()).episode_num == args.batch_size/2 for x in agent_list]):
                    array_list = [ray.get(x.get_buffer.remote()).numpy_() for x in agent_list]
                    collective = [[x[i][:int(args.batch_size/len(agent_list))] for x in array_list] for i in array_list[0].keys()]
                    collective_buffer = [np.concatenate(x) for x in collective]
                    
                    #print('collective_buffer',collective_buffer)
                    for i in range(len(agent_list)):
                        agent_list[i].add_to_buffer.remote(collective_buffer)
           
                
                output = [agent_list[i].act.remote(state_list[i], i_epoch, args.lvl) for i in range(len(agent_list))]
                result = ray.get(output)
                
                for c in range(len(agent_list)):
                    state_list[c] = result[c][0]
                    if args.lvl == 1:
                        a[state_list[c][0], state_list[c][2]] -= 1
                        a[state_list[c][1], state_list[c][3]] -= 1
                        score_list[c] += result[c][1]
                        
                    else:
                        b[state_list[c][0], state_list[c][1]] -= 1
                        done_list[c] = result[c][1]
                        
                
                if args.lvl == 1:
                    
                    best_score_index = score_list.index(max(score_list)) 
                


                

                if args.lvl == 1 and max(score_list) == 2 and len(agent_list) < 4:
                    print('All food collected! next agent joining...','collecter: Agent',best_score_index)
                    
                    agent_list.append(ray.get(agent_list[best_score_index].clone.remote(max_id+1)))
                    state_list.append(state_list[best_score_index])
                    score_list.append(0)
                    break

                
                
                if all(done_list) and args.lvl == 2:
                    
                    print('LEVEL 2 ACHIEVED : All places taken by separate Agent -- juhhuuuuu')
                    
                    break
            
            evolution_reward = [evolution_reward[i] + np.sum(reward_list[i]) for i in range(len(agent_list))]   
            evolution_loss = [evolution_loss[i] + ray.get(agent_list[i].get_losses.remote())['actor_loss'] + ray.get(agent_list[i].get_losses.remote())['critic_loss'] + 
                              ray.get(agent_list[i].get_losses.remote())['entropy_loss'] for i in range(len(agent_list))]    
            print(a)
            log_count += 1

        for i in range(len(agent_list)):
            writer.add_scalar('Total Evolution losses agents {}'.format(i),evolution_loss[i] ,i_evolution)
            writer.add_scalar('Total Evolution rewards agents {}'.format(i),evolution_reward[i],i_evolution)
            writer.add_scalar('Total Evolution state agents {}'.format(i),result_reproduction[i][0],i_evolution)
            writer.add_scalar('Total Evolution actions agents {}'.format(i),result_reproduction[i][1],i_evolution)
            
        update_mask = ray.get([agent_list[i].reproduction_rewarding.remote(result_reproduction[i][0], 
                            result_reproduction[i][1], result_reproduction[i][2], evolution_reward[i], result_reproduction[i][3], evolution_loss[i], i_evolution) for i in range(len(agent_list))])
        print(update_mask)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=128, help=" Maximum number of training steps")
    parser.add_argument("--evolution_lenght", type=int, default=1000, help=" Maximum number of evolution steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--agent_number", type=int, default=2, help="Number of agents")
    parser.add_argument("--policy_dist", type=str, default="Discrete RNN", help="Discrete or Discrete RNN or Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--reprod_batch_size", type=int, default=3, help="Batch size for reproduction network")
    parser.add_argument("--reprod_mini_batch_size", type=int, default=2, help="Minibatch size for reproduction network")
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden dimension of the RNN")
    parser.add_argument("--n_layers", type=int, default=3, help="The number of layers in RNN")
    parser.add_argument("--lr_a", type=float, default=3e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-3, help="Learning rate of critic")
    parser.add_argument("--lr_r", type=float, default=3e-3, help="Learning rate of reproducer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--target_kl", type=float, default=0.9, help="KL Divergence target for early stopping")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.001, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_gru", type=float, default=True, help="Gated Recurrent Unit")
    parser.add_argument("--use_collective", type=bool, default=False, help="Sharing memory across agents")
    parser.add_argument("--collective_switch", type=int, default=8, help="The number of epoch to switch to collective learning")
    parser.add_argument("--use_shuffle", type=bool, default=False, help="Shuffle updating order")
    args = parser.parse_args()

    env_name = ['Environment Petri Dish']
    env_index = 1
    main(args, env_name=env_name[0], number=4_4)
    #args.use_collective = False
    #main(args, env_name=env_name[0], number=1_25)