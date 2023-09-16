import numpy as np
import torch

from ppo import utils
from ppo.envs import make_vec_envs


def evaluate(actor_critic, obs_rms=None, normalize=True, env_name='NavEnv-v0', seed=None, num_processes=1,
             device=torch.device('cpu'), ret_info=1, capture_video=False, env_kwargs={}, data_callback=None,
             num_episodes=10, verbose=0, with_activations=False, deterministic=True,
             aux_wrapper_kwargs={}, auxiliary_truth_sizes=[],
             eval_log_dir=None, video_folder='./video'):
    '''
    ret_info: level of info that should be tracked and returned
    capture_video: whether video should be captured for episodes
    env_kwargs: any kwargs to create environment with
    data_callback: a function that should be called at each step to pull information
        from the environment if needed. The function will take arguments
            def callback(actor_critic, vec_envs, recurrent_hidden_states, data):
        actor_critic: the actor_critic network
        vec_envs: the vec envs (can call for example vec_envs.get_attr('objects') to pull data)
        recurrent_hidden_states: these are given in all data, but may want to use in computation
        obs: observation this step (after taking action) - 
            note that initial observation is never seen by data_callback
            also note that this observation will have the mean normalized
            so may instead want to call vec_envs.get_method('get_observation')
        action: actions this step
        reward: reward this step
        data: a data dictionary that will continuously be passed to be updated each step
            it will start as an empty dicionary, so keys must be initialized
        see below at example_data_callback in this file for an example
    '''

    if seed is None:
        seed = np.random.randint(0, 1e9)

    envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, 
                              capture_video=capture_video, 
                              env_kwargs=env_kwargs, normalize=normalize,
                              video_folder=video_folder,
                              **aux_wrapper_kwargs)

    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    all_obs = []
    all_actions = []
    all_rewards = []
    all_rnn_hxs = []
    all_dones = []
    all_masks = []
    all_activations = []
    all_values = []
    all_actor_features = []
    all_auxiliary_preds = []
    all_auxiliary_truths = []
    data = {}
    
    ep_obs = []
    ep_actions = []
    ep_rewards = []
    ep_rnn_hxs = []
    ep_dones = []
    ep_values = []
    ep_masks = []
    ep_actor_features = []
    
    ep_auxiliary_preds = []
    ep_activations = []
    ep_auxiliary_truths = []
    

    obs = envs.reset()
    rnn_hxs = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)

    for i in range(num_episodes):
        step = 0
        
        while True:
            ep_obs.append(obs)
            ep_rnn_hxs.append(rnn_hxs)
            if data_callback is not None and step == 0:
                data = data_callback(None, envs, rnn_hxs,
                    obs, [], [], [False], data, first=True)

            with torch.no_grad():
                outputs = actor_critic.act(obs, rnn_hxs, 
                                        masks, deterministic=deterministic,
                                        with_activations=with_activations)
                action = outputs['action']
                rnn_hxs = outputs['rnn_hxs']
            obs, reward, done, infos = envs.step(action)
            
            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_values.append(outputs['value'])
            ep_masks.append(masks)
            ep_actor_features.append(outputs['actor_features'])
            
            if 'auxiliary_preds' in outputs:
                ep_auxiliary_preds.append(outputs['auxiliary_preds'])
            
            if with_activations:
                ep_activations.append(outputs['activations'])

            if data_callback is not None:
                data = data_callback(None, envs, rnn_hxs,
                    obs, action, reward, done, data)
            else:
                data = {}
                
            auxiliary_truths = [[] for i in range(len(actor_critic.auxiliary_output_sizes))]
            for info in infos:
                if 'auxiliary' in info and len(info['auxiliary']) > 0:
                    for i, aux in enumerate(info['auxiliary']):
                        auxiliary_truths[i].append(aux)
            if len(auxiliary_truths) > 0:
                auxiliary_truths = [torch.tensor(np.vstack(aux)) for aux in auxiliary_truths]
            ep_auxiliary_truths.append(auxiliary_truths)
            
            
            # for info in infos:
            #     if 'episode' in info.keys():
            #         eval_episode_rewards.append(info['episode']['r'])
            #         #Andy: add verbosity option
            #         if verbose >= 2:
            #             print('ep ' + str(len(eval_episode_rewards)) + ' rew ' + \
            #                 str(info['episode']['r']))
            
            step += 1
            
            if done[0]:
                all_obs.append(np.vstack(ep_obs))
                all_actions.append(np.vstack(ep_actions))
                all_rewards.append(np.vstack(ep_rewards))
                all_rnn_hxs.append(np.vstack(ep_rnn_hxs))
                all_dones.append(np.vstack(ep_dones))
                all_masks.append(np.vstack(ep_masks))
                all_values.append(np.vstack(ep_values))
                all_actor_features.append(np.vstack(ep_actor_features))
                
                all_auxiliary_preds.append(ep_auxiliary_preds)
                all_activations.append(ep_activations)
                all_auxiliary_truths.append(ep_auxiliary_truths)

                if data_callback is not None:
                    data = data_callback(None, envs, rnn_hxs,
                        obs, action, reward, done, data, stack=True)
                          
                if verbose >= 2:
                    print(f'ep {i}, rew {np.sum(ep_rewards)}' )
                    
                ep_obs = []
                ep_actions = []
                ep_rewards = []
                ep_rnn_hxs = []
                ep_dones = []
                ep_values = []
                ep_masks = []
                ep_actor_features = []
                
                ep_auxiliary_preds = []
                ep_activations = []
                ep_auxiliary_truths = []
                
                step = 0
                
                break
            
            
  

    envs.close()
    if verbose >= 1:
        print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    return {
        'obs': all_obs,
        'actions': all_actions,
        'rewards': all_rewards,
        'rnn_hxs': all_rnn_hxs,
        'dones': all_dones,
        'masks': all_masks,
        'envs': envs,
        'data': data,
        'activations': all_activations,
        'values': all_values,
        'actor_features': all_actor_features,
        'auxiliary_preds': all_auxiliary_preds,
        'auxiliary_truths': all_auxiliary_truths,
    }


def example_data_callback(actor_critic, vec_envs, recurrent_hidden_states, data):
    pass



def nav_data_callback(agent, env, rnn_hxs, obs, action, reward, done, data, stack=False,
                      first=False):
    '''
    Add navigation data pos and angle to data object
    If stack is True, this function will handle stacking the data properly
    '''    
    if 'pos' not in data:
        data['pos'] = []
    if 'angle' not in data:
        data['angle'] = []
    if 'ep_pos' not in data:
        data['ep_pos'] = []
    if 'ep_angle' not in data:
        data['ep_angle'] = []

    if stack:
        data['pos'].append(np.vstack(data['ep_pos']))
        data['angle'].append(np.vstack(data['ep_angle']))
        
        data['ep_pos'] = []
        data['ep_angle'] = []    
    elif not done[0]:
        pos = env.get_attr('character')[0].pos.copy()
        angle = env.get_attr('character')[0].angle
        data['ep_pos'].append(pos)
        data['ep_angle'].append(angle)
    
    return data


def explore_data_callback(agent, env, rnn_hxs, obs, action, reward, done, data, stack=False,
                      first=False):
    '''
    Add navigation data pos and angle and position of platform to data
    '''
    if 'pos' not in data:
        data['pos'] = []
    if 'angle' not in data:
        data['angle'] = []
    if 'goal' not in data:
        data['goal'] = []
    if 'ep_pos' not in data:
        data['ep_pos'] = []
    if 'ep_angle' not in data:
        data['ep_angle'] = []
    if 'ep_goal' not in data:
        data['ep_goal'] = []
        
    if 'goal_reached' not in data:
        data['goal_reached'] = []
    if 'ep_goal_reached' not in data:
        data['ep_goal_reached'] = []

    if first:
        data['ep_goal'].append(env.envs[0].boxes[-1].corner)

    if stack:
        data['pos'].append(np.vstack(data['ep_pos']))
        data['angle'].append(np.vstack(data['ep_angle']))
        data['goal'].append(data['ep_goal'])
        data['goal_reached'].append(data['ep_goal_reached'])
        
        data['ep_pos'] = []
        data['ep_angle'] = []        
        data['ep_goal'] = []
        data['ep_goal_reached'] = []
    elif not done[0]:
        pos = env.get_attr('character')[0].pos.copy()
        angle = env.get_attr('character')[0].angle
        goal_reached = env.get_attr('goal_reached_this_ep')[0]
        data['ep_pos'].append(pos)
        data['ep_angle'].append(angle)
        data['ep_goal_reached'].append(goal_reached)
    
    return data


def shortcut_data_callback(agent, env, rnn_hxs, obs, action, reward, done, data, stack=False,
                      first=False):
    '''
    Add navigation data pos and angle and position of platform to data
    '''
    if 'pos' not in data:
        data['pos'] = []
    if 'angle' not in data:
        data['angle'] = []
    if 'shortcut' not in data:
        data['shortcut'] = []
    if 'ep_pos' not in data:
        data['ep_pos'] = []
    if 'ep_angle' not in data:
        data['ep_angle'] = []
    if 'ep_shortcut' not in data:
        data['ep_shortcut'] = []

    if first:
        data['ep_shortcut'] = env.envs[0].shortcuts_available

    if stack:
        data['pos'].append(np.vstack(data['ep_pos']))
        data['angle'].append(np.vstack(data['ep_angle']))
        data['shortcut'].append(data['ep_shortcut'])
        
        data['ep_pos'] = []
        data['ep_angle'] = []        
        data['ep_shortcut'] = []
    elif not done[0]:
        pos = env.get_attr('character')[0].pos.copy()
        angle = env.get_attr('character')[0].angle
        data['ep_pos'].append(pos)
        data['ep_angle'].append(angle)
    
    return data

def simple_vec_envs(obs_rms=None, env_name='NavEnv-v0', normalize=True, seed=None, num_processes=1,
             device=torch.device('cpu'), capture_video=False, env_kwargs={},
             aux_wrapper_kwargs={}, eval_log_dir=None):
    if seed is None:
        seed = np.random.randint(0, 1e9)

    envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, 
                              capture_video=capture_video, 
                              env_kwargs=env_kwargs, normalize=normalize,
                              **aux_wrapper_kwargs)
    
    vec_norm = utils.get_vec_normalize(envs)
    
    if obs_rms is None:
        #If obs_rms is not given, make it a learning normalize vector envs
        pass
    else:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms
        
    return envs



def evaluate_steps(actor_critic, envs, num_steps=10, data_callback=None, 
                   device=torch.device('cpu'), deterministic=True,
                   with_activations=False):
    data = {}
    ep_obs = []
    ep_actions = []
    ep_rewards = []
    ep_rnn_hxs = []
    ep_dones = []
    ep_values = []
    ep_masks = []
    ep_actor_features = []
    
    ep_auxiliary_preds = []
    ep_activations = []
    ep_auxiliary_truths = []
    

    num_processes = len(envs.envs)
    obs = envs.reset()
    rnn_hxs = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)

    step = 0
    
    while step < num_steps:
        ep_obs.append(obs)
        ep_rnn_hxs.append(rnn_hxs)
        if data_callback is not None and step == 0:
            data = data_callback(None, envs, rnn_hxs,
                obs, [], [], [False], data, first=True)

        with torch.no_grad():
            outputs = actor_critic.act(obs, rnn_hxs, 
                                    masks, deterministic=deterministic,
                                    with_activations=with_activations)
            action = outputs['action']
            rnn_hxs = outputs['rnn_hxs']
        obs, reward, done, infos = envs.step(action)
        
        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)
        
        ep_actions.append(action)
        ep_rewards.append(reward)
        ep_dones.append(done)
        ep_values.append(outputs['value'])
        ep_masks.append(masks)
        ep_actor_features.append(outputs['actor_features'])
        
        if 'auxiliary_preds' in outputs:
            ep_auxiliary_preds.append(outputs['auxiliary_preds'])
        
        if with_activations:
            ep_activations.append(outputs['activations'])

        if data_callback is not None:
            data = data_callback(None, envs, rnn_hxs,
                obs, action, reward, done, data)
        else:
            data = {}
            
        auxiliary_truths = [[] for i in range(len(actor_critic.auxiliary_output_sizes))]
        for info in infos:
            if 'auxiliary' in info and len(info['auxiliary']) > 0:
                for i, aux in enumerate(info['auxiliary']):
                    auxiliary_truths[i].append(aux)
        if len(auxiliary_truths) > 0:
            auxiliary_truths = [torch.tensor(np.vstack(aux)) for aux in auxiliary_truths]
        ep_auxiliary_truths.append(auxiliary_truths)
        
        
        # for info in infos:
        #     if 'episode' in info.keys():
        #         eval_episode_rewards.append(info['episode']['r'])
        #         #Andy: add verbosity option
        #         if verbose >= 2:
        #             print('ep ' + str(len(eval_episode_rewards)) + ' rew ' + \
        #                 str(info['episode']['r']))
        
        step += 1
        
        if done[0]:
            break
    
    return {
        'obs': ep_obs,
        'actions': ep_actions,
        'rewards': ep_rewards,
        'rnn_hxs': ep_rnn_hxs,
        'dones': ep_dones,
        'masks': ep_masks,
        'envs': envs,
        'data': data,
        'activations': ep_activations,
        'values': ep_values,
        'actor_features': ep_actor_features,
        'auxiliary_preds': ep_auxiliary_preds,
        'auxiliary_truths': ep_auxiliary_truths,
    }
