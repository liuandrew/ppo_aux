import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir,
             device, ret_info=1, capture_video=False, env_kwargs={}, data_callback=None,
             num_episodes=10, verbose=0, with_activations=False, deterministic=True,
             normalize=True, aux_wrapper_kwargs={}, new_aux=False, auxiliary_truth_sizes=[]):
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

    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, 
                              capture_video=capture_video, 
                              env_kwargs=env_kwargs, normalize=normalize,
                              **aux_wrapper_kwargs)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    all_obs = []
    all_actions = []
    all_rewards = []
    all_hidden_states = []
    all_dones = []
    all_masks = []
    all_activations = []
    all_values = []
    all_actor_features = []
    all_auxiliary_preds = []
    all_auxiliary_truths = []
    data = {}

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            # _, action, _, eval_recurrent_hidden_states, _ = actor_critic.act(
            #     obs,
            #     eval_recurrent_hidden_states,
            #     eval_masks,
            #     deterministic=True)
            
            outputs = actor_critic.act(obs, eval_recurrent_hidden_states, 
                                       eval_masks, deterministic=deterministic,
                                       with_activations=with_activations)
            action = outputs['action']
            eval_recurrent_hidden_states = outputs['rnn_hxs']
            
        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)
        
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        all_obs.append(obs)
        all_actions.append(action)
        all_rewards.append(reward)
        all_hidden_states.append(eval_recurrent_hidden_states)
        all_dones.append(done)
        all_masks.append(eval_masks)
        all_values.append(outputs['value'])
        all_actor_features.append(outputs['actor_features'])
        
        if 'auxiliary_preds' in outputs:
            all_auxiliary_preds.append(outputs['auxiliary_preds'])
        
        
        if with_activations:
            all_activations.append(outputs['activations'])

        if data_callback is not None:
            data = data_callback(actor_critic, eval_envs, eval_recurrent_hidden_states,
                obs, action, reward, done, data)
        else:
            data = {}
        if new_aux:
            auxiliary_truths = [[] for i in range(len(actor_critic.auxiliary_output_sizes))]
            for info in infos:
                if 'auxiliary' in info and len(info['auxiliary']) > 0:
                    for i, aux in enumerate(info['auxiliary']):
                        auxiliary_truths[i].append(aux)
            if len(auxiliary_truths) > 0:
                auxiliary_truths = [torch.tensor(np.vstack(aux)) for aux in auxiliary_truths]

        else:
            auxiliary_truths = []
            for info in infos:
                if 'auxiliary' in info:
                        if len(info['auxiliary'] > 0):
                            auxiliary_truths.append(info['auxiliary'])
                        
            if len(auxiliary_truths) > 0:
                auxiliary_truths = torch.tensor(np.vstack(auxiliary_truths))
            else:
                auxiliary_truths = None
        
        all_auxiliary_truths.append(auxiliary_truths)
        
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                #Andy: add verbosity option
                if verbose >= 2:
                    print('ep ' + str(len(eval_episode_rewards)) + ' rew ' + \
                        str(info['episode']['r']))
                        

    # eval_envs.close()
    if verbose >= 1:
        print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    return {
        'obs': all_obs,
        'actions': all_actions,
        'rewards': all_rewards,
        'hidden_states': all_hidden_states,
        'dones': all_dones,
        'masks': all_masks,
        'envs': eval_envs,
        'data': data,
        'activations': all_activations,
        'values': all_values,
        'actor_features': all_actor_features,
        'auxiliary_preds': all_auxiliary_preds,
        'auxiliary_truths': all_auxiliary_truths,
    }


def example_data_callback(actor_critic, vec_envs, recurrent_hidden_states, data):
    pass