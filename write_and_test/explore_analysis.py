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
    
    
'''

Search efficiency testing and recording activations during explore trajectories

'''    
    
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




'''
This is how grid points and grid angles are generated for explore activation collection
But in case the seeding for angles doesn't work, we save them to be consistent
'''
# np.random.seed(0)
# a = np.linspace(5, 295, 6)
# x, y = np.meshgrid(a, a)
# grid_points = np.array([x.reshape(-1), y.reshape(-1)]).T
# grid_angles = np.random.uniform(-np.pi, np.pi, size=len(grid_points))


# small set of trajectories, a bit of angle inconsistency in the middle
# copy_key = ['batch128_64', 'batch128_128', '_64']
# copy_t = [0, 1, 0]

# copy_key = ['batch128_64', 'batch128_64', 'batch128_128', '_64', '_128', '_128']
# copy_t = [0, 2, 1, 0, 0, 1]


'''
Final - uncomment if using
'''
# grid_points, grid_angles = pickle.load(open('data/explore/grid_starts', 'rb'))
# actions = pickle.load(open('data/explore/copy_explore_actions', 'rb'))

# # fairly evened out but one long angle bar diagonal
# copy_key = ['batch128_64', 'batch128_128', '_64', '_128', '_128']
# copy_t = [0, 1, 0, 0, 1]


def test_forced_search_activations(model, obs_rms, env_kwargs, save=None):
    '''
    Run search tests for some selected copied actions
    
    save: optionally pass a file name to save the results as
    '''
    all_eff, all_traj, all_res = [], [], []
    for key, t in tqdm(zip(copy_key, copy_t), total=len(copy_key)):
        eff, traj, res = test_search_efficiency(model, obs_rms, env_kwargs=env_kwargs,
                                            manual_starts=[grid_points, grid_angles],
                                            forced_actions=actions[key][t])
        all_eff += eff
        all_traj += traj
        all_res += res
        
    res = combine_evaluation_results(all_res)
    res['activations'] = ep_stack_activations(res, combine=True)
    del res['envs']
        
    if save is not None:            
        pickle.dump(res, open(save, 'wb'))
        
    return all_eff, all_traj, res


def test_search_activations(model, obs_rms, env_kwargs, save=None):
    '''
    Run search tests for some selected copied actions
    
    save: optionally pass a file name to save the results as
    '''
    effs, trajs, ress = test_search_efficiency(model, obs_rms, env_kwargs=env_kwargs, ret_res=True,
                        manual_starts=[grid_points, grid_angles])
        
    res = combine_evaluation_results(ress)
    res['activations'] = ep_stack_activations(res, combine=True)
    del res['envs']
        
    if save is not None:            
        pickle.dump(res, open(save, 'wb'))
        
    return effs, trajs, res




def combine_evaluation_results(ress):
    '''
    Combine multiple appended results from multiple evaluate calls
    Ex.
    _, _, ress = test_search_efficiency(model, obs_rms, ...)
    res = combine_evaluation_results(ress)
    '''
    res = {}
    for key in ress[0]:
        if key == 'data':
            res[key] = {subkey: [] for subkey in ress[0][key]}
        elif key == 'envs':
            continue
        else:
            res[key] = []
    
    
    for r in ress:
        
        for key in r:
            if key == 'data':
                for subkey in r[key]:
                    res[key][subkey] += r[key][subkey]
            elif key == 'envs':
                continue
            else:
                res[key] += r[key]
    
    res['envs'] = ress[0]['envs']
    return res
    
    
    
    
# def ep_stack_activations(res, combine=False):
#     '''Stack activations dictionaries from episodic evaluation calls
    
#     combine: whether to combine all activations so there are no episodic separations
#     returns: list of dictionaries
#         Ex. activs[ep]['shared_activations'][layer_num]
#     '''    
#     activs = []
#     for ep in range(len(res['activations'])):
#         activs.append(stack_activations(res['activations'][ep]))
        
#     if combine:
#         stacked_activs = {}
#         for key in activs[0]:
#             stacked_activs[key] = torch.hstack(
#                 [activs[ep][key] for ep in range(len(activs))])
#         activs = stacked_activs
#     return activs




def hm_process_res(res, activ_key='shared_activations', activ_layer=1,
                   angle_mod=True):
    '''
    Process list of res (ress) that come from a test_search_efficiency or
        test_forced_search_activations function call to generate heatmaps
        
    activ_key: 'shared_activations'/'actor_activations'/'critic_activations'
    activ_layer: which number layer of the activation type
    angle_mod: whether to mod out the direction contribution of activation
        from the spatial heatmap
    '''
    # res = combine_evaluation_results(ress)
    # activs = ep_stack_activations(res, combine=True)
    activs = res['activations']
    pos = np.vstack(res['data']['pos'])
    angle = np.vstack(res['data']['angle']).squeeze()

    num_nodes = activs[activ_key].shape[2]
    
    # sigma=50 to get smooth contribution of direction in activations
    if angle_mod:
        mod_angle_hms = []
        for i in range(num_nodes):
            activ = activs[activ_key][activ_layer, :, i]
            uniform_angles, weights = circular_gaussian_filter_fixed_angles(angle.squeeze(), activ,  sigma=50)
            mod_angle_hms.append(weights)
    
    # sigma=5 for these to detect structure in direction maps
    structure_angle_hms = []
    for i in range(num_nodes):
        activ = activs[activ_key][activ_layer, :, i]
        uniform_angles, weights = circular_gaussian_filter_fixed_angles(angle.squeeze(), activ,  sigma=5)
        structure_angle_hms.append(weights)

    close_angle_idxs = np.argmin(np.abs(uniform_angles.reshape(-1, 1) - angle), axis=0)
    spatial_hms = []
    for i in range(num_nodes):
        if angle_mod:
            activ = activs[activ_key][activ_layer, :, i] - mod_angle_hms[i][close_angle_idxs]
        else:
            activ = activs[activ_key][activ_layer, :, i]
        hm = gaussian_smooth(pos, activ)
        spatial_hms.append(hm)
        
    return spatial_hms, structure_angle_hms, res
    
    

    




def gaussian_smooth(pos, y, extent=(5, 295), num_grid=30, sigma=10,
                    ret_hasval=False):
    """Convert a list of positions and values to a smoothed heatmap

    Args:
        pos (): _description_
        y (_type_): _description_
        extent (tuple, optional): _description_. Defaults to (5, 295).
        num_grid (int, optional): _description_. Defaults to 30.
        sigma (int, optional): _description_. Defaults to 10.
        ret_hasval (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # a = stacked['shared_activations'][0, :, 0].numpy()
    y = np.array(y)
    
    grid = np.linspace(extent[0], extent[1], num_grid)
    xs, ys = np.meshgrid(grid, grid)
    ys = ys[::-1]
    smoothed = np.zeros(xs.shape)
    hasval = np.zeros(xs.shape)
    for i in range(num_grid):
        for j in range(num_grid):
            p = np.array([xs[i, j], ys[i, j]])
            dists = np.sqrt(np.sum((pos - p)**2, axis=1))
            g = np.exp(-dists**2 / (2*sigma**2))
            
            if len(g[g > 0.1]) < 1:
                val = 0
            else:
                val = np.sum(y[g > 0.1] * g[g > 0.1]) / np.sum(g[g > 0.1])
                hasval[i, j] = 1

            smoothed[i, j] = val
    if ret_hasval:
        return smoothed, hasval
    else:
        return smoothed
    
    
    
def circular_gaussian_filter_fixed_angles(angles, weights, sigma=3, num_angles=100):
    '''Gaussian filter across angles
    
    angles: array of angles
    weights: activations at each of the time points corresponding to angles
    sigma: smoothing constant
        Use around 5 to probe for structure
        Around 50 seems to work to create heatmaps used to mod out direction activation
            for spatial heatmaps
    '''
    # Sort angles and corresponding weights
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Wrap the angles and weights around the circle to handle boundary cases
    wrapped_angles = np.concatenate((wrap_angle(sorted_angles - 2 * np.pi), sorted_angles, wrap_angle(sorted_angles + 2 * np.pi)))
    wrapped_weights = np.concatenate((sorted_weights, sorted_weights, sorted_weights))

    # Apply Gaussian filter
    filtered_weights = gaussian_filter1d(wrapped_weights, sigma)

    # Extract the filtered weights corresponding to the original angles
    filtered_weights = filtered_weights[len(angles): 2 * len(angles)]

    # Create uniformly spaced angles
    uniform_angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)

    # Interpolate the filtered weights for the uniformly spaced angles
    uniform_filtered_weights = np.interp(uniform_angles, sorted_angles, filtered_weights)

    return uniform_angles, uniform_filtered_weights
    
    
    
def get_explore_kwargs(obs_set=3, max_steps=300, shaping='none'):
    '''
    Typical env_kwargs used in exploration
    obs_set: 1/2/3 for vision_only/vision+location/vision+location+last_goal_pos
    max_steps: 300 for search testing, 500 for actual episodes
    shaping: 
        'none': only goal
        'bonus': goal+bonus
        'punish': goal+punish
        'bonus_punish': goal+bonus+punish
    '''
    env_kwargs = {'obs_set': obs_set,
                  'max_steps': max_steps}
    
    if 'bonus' in shaping and 'punish' in shaping:
        env_kwargs['rew_structure'] = 'explorepunish1_explorebonus'
        env_kwargs['sub_goal_reward'] = 0.04
        env_kwargs['bonus_multiplier'] = 5
        env_kwargs['explore_punish_arg'] = 5
    elif 'bonus' in shaping:
        env_kwargs['rew_structure'] = 'explorebonus'
        env_kwargs['sub_goal_reward'] = 0.04
        env_kwargs['bonus_multiplier'] = 5
    elif 'punish' in shaping:
        env_kwargs['rew_structure'] = 'explorepunish1'
        env_kwargs['sub_goal_reward'] = 0.04
        env_kwargs['explore_punish_arg'] = 5
    
    return env_kwargs
        
    
    
