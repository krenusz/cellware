


#env = Environment 
import ray
ray.init(ignore_reinit_error=True,address='auto', runtime_env={"working_dir": "C:/Users/User/.conda/envs/mtgat"})

@ray.remote
def cells1(env, agent, state, i_epoch, cnt):
    from threadpoolctl import threadpool_limits
    #from environment_dish import Environment
    from collections import namedtuple

    Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])
    
    
    with threadpool_limits(limits=1, user_api='blas'):
        
        action, action_prob = agent.select_action(state)
        print('action',action,'state',state)

        next_state, reward, _, _, food1_loc, food2_loc, food3_loc, _, food1_collected, food2_collected, food3_collected = env.step(action,state)
        trans = Transition(state, action, action_prob, reward, next_state)

        if all(next_state == food1_loc) and reward >= 100:
            score = 1
            print('food1 collected')
        elif all(next_state == food2_loc) and reward >= 100:
            score = 1
            print('food2 collected')
        elif all(next_state == food3_loc) and reward >= 100:
            score = 1
            print('food3 collected')
        else:
            score = 0
        agent.store_transition(trans)

        if cnt == 150: 
            print('done: agent getting updated')
            if len(agent.buffer) >= agent.batch_size:
                agent.update(i_epoch)

    return env, agent, next_state, score