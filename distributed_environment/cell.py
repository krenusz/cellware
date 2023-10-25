

import ray
ray.init(ignore_reinit_error=True,address='auto', runtime_env={"working_dir": "C:/Users/User/.conda/envs/mtgat"})

@ray.remote
def cells1(env, agent, state, i_epoch, cnt, lvl, done):
    from threadpoolctl import threadpool_limits
    #from environment_dish import Environment
    from collections import namedtuple

    Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])
    
    
    with threadpool_limits(limits=1, user_api='blas'):
        score = 0
        
        if done:
            agent.update(i_epoch) # update and clear experience
            return env, agent, state, done
        
        action, action_prob = agent.select_action(state)
        print('action',action,'state',state)

        next_state, reward_lvl1, reward_lvl2, _, _, food1_loc, food2_loc, food3_loc, _, food1_collected, food2_collected, food3_collected = env.step(action,state)
        
        if lvl == 1:
            reward = reward_lvl1
        else:
            reward = reward_lvl2
        print('REWARD:',reward)
        trans = Transition(state, action, action_prob, reward, next_state)

        #if all(next_state == food1_loc) and reward >= 100:
        #    score = 1
        #    print('food1 collected')
        #elif all(next_state == food2_loc) and reward >= 100:
        #    score = 1
        #    print('food2 collected')
        #elif all(next_state == food3_loc) and reward >= 100:
        #    score = 1
        #    print('food3 collected')
        if reward == 90 and lvl == 1:
            score = 1
            print('CONDITION: food collected')
        elif reward == 90 and lvl == 2:
            done = True
            
            print('place taken')

        elif reward != 90 and lvl == 1:
            score = 0
            print('CONDITION: food not collected')
        elif reward != 9 and lvl == 2:
            done = False
        agent.store_transition(trans)

        
        
        if len(agent.buffer) >= agent.batch_size:
            print('done: agent getting updated')
            agent.update(i_epoch)
    if lvl == 1:
        return env, agent, next_state, score
    else:
        return env, agent, next_state, done
