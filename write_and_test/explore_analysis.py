import gym
import torch
import gym_nav
import numpy as np

from evaluation import *
from model_evaluation import *
from trajectories import *


def split_by_rew(targets_to_split, rews):
    '''
    Used to find when in an explore task the goal was reached based on
    when reward was earned. 

    For example, 
    res = evalu(...)
    ep_pos = split_by_ep(res['data']['pos'], res['rewards'])
    '''
    split_idxs = np.where(np.vstack(rews == 1))[0] + 1
    
    # if nothing to split
    if len(split_idxs) == 0:
        return [targets_to_split]

    split_targets = []
    for i in range(len(split_idxs)):
        if i == 0:
            done_targets = targets_to_split[:split_idxs[i]]
        else:
            done_targets = targets_to_split[split_idxs[i-1]:split_idxs[i]]
        split_targets.append(done_targets)
        
    if split_idxs[-1] != len(rews):
        split_targets.append(targets_to_split[split_idxs[-1]:])
        
    return split_targets


def compute_search_efficiency(pos, covered_distance=10):
    '''
    Compute search efficiency based on how many spots agent managed to get close to
    
    covered_distance: how close the agent had to get to a point to count as covered
        this should probably related to speed agent moves and goal size?
    '''
    
    test_points = np.stack(np.meshgrid(np.linspace(0, 300, 301), np.linspace(0, 300, 301))).reshape(2, -1).T
    unique_pos = np.unique(pos, axis=0)
    
    dists = []
    for p in unique_pos:
        dists.append(np.sqrt(np.sum((test_points - p) ** 2, axis=1)))
    min_dists = np.vstack(dists).min(axis=0)
    
    color_points = np.full(test_points.shape, False)
    color_points[min_dists < 10] = True
    color_points = color_points.all(axis=1).reshape(301, 301) #this can be plt.imshow'd to see what counts as visited
    num_covered_points = color_points.sum()
    
    total_points = 301*301
    cover_efficiency = num_covered_points / total_points / pos.shape[0]
    
    return cover_efficiency
    
    
def compute_eps_search_efficiency(all_pos, all_rew, ret_trials=False):
    '''
    Compute the mean and std search efficiency of an agent across multiple episodes
    Note we are assuming that all_pos and all_rew are already split by episode which
        they should be coming from an evaluate() call
        
    ret_trials: return individual values rather than mean and std, for example for boxplots    
    
    Ex. 
    res = evaluate(model, obs_rms, env_kwargs=env_kwargs, num_episodes=5, data_callback=nav_data_callback)
    compute_eps_search_efficiency(res['data']['pos'], res['rewards']
    '''
    
    explore_efficiencies = []

    for ep in range(len(all_rew)):
        rew = all_rew[ep]
        pos = all_pos[ep]

        split_pos = split_by_rew(pos, rew)
        split_rew = split_by_rew(rew, rew)

        for p in split_pos:
            explore_efficiencies.append(compute_search_efficiency(p))
            
    mean_eff = np.mean(explore_efficiencies)
    std_eff = np.std(explore_efficiencies)
    
    if ret_trials:
        return explore_efficiencies
    else:
        return mean_eff, std_eff
    
    
def test_search_efficiency(model, obs_rms, test_set=2, env_kwargs={}, ret_res=True,
                          manual_starts=None, forced_actions=None, with_activations=True):
    '''
    Test an agent's exploring efficiency given fixed starting points
    
    env_kwargs: extra arguments to give. Note: these overwrite the kw appended, which are
        'goal_size' and 'fixed_reset', so ideally do not have these as part of passed env_kwargs
    manual_starts: option to pass [reset_points, reset_angles] to use manually
    
    forced_actions: optional list of actions. Should have same number of episode actions
        as number of start pointsx
    '''
    if test_set == 0:
        reset_points = [np.array([150., 150.])]
        reset_angles = [np.pi/2]
    
    elif test_set == 1:
        reset_points = [np.array([150., 150.]),
                        np.array([10., 290.])]
        reset_angles = [np.pi/2,
                        0.]
    
    elif test_set == 2:
        reset_points = [np.array([150., 150.]),
                        np.array([10., 290.]),
                        np.array([290., 290.]),
                        np.array([290., 10.]),
                        np.array([10., 10.])]
        reset_angles = [np.pi/2,
                        0.,
                        -np.pi/2,
                        np.pi/2,
                        0.]
        
    if manual_starts is not None:
        reset_points = manual_starts[0]
        reset_angles = manual_starts[1]
        
    effs = []
    trajs = []
    ress = []
    for i, (point, angle) in enumerate(zip(reset_points, reset_angles)):
        kw = {'goal_size': 1e-8, 'fixed_reset': [point, angle]}
        for k in env_kwargs:
            kw[k] = env_kwargs[k]
        
        if forced_actions is not None:
            res = forced_action_evaluate(model, obs_rms, env_kwargs=kw,
                        env_name='ExploreNav-v0', num_episodes=1, forced_actions=forced_actions[i],
                        data_callback=explore_data_callback, with_activations=with_activations)
        else:
            res = evaluate(model, obs_rms, env_kwargs=kw,
                        env_name='ExploreNav-v0', num_episodes=1, 
                        data_callback=explore_data_callback, with_activations=with_activations)

        eff, _ = compute_eps_search_efficiency(res['data']['pos'], res['rewards'])
        effs.append(eff)
        trajs.append((res['data']['pos'][0], res['data']['angle'][0]))
        ress.append(res)
        
    return effs, trajs, ress

    


    
    