import numpy as np
from evaluation import *
from model_evaluation import *
from umap import UMAP
from explore_analysis import ep_stack_activations
from tqdm import tqdm
import proplot as pplt

turn_speed = 0.3
move_speed = 10

shortcut_target = np.array([150, 250])
path_target = np.array([25, 250])
goal_target = np.array([275, 275])

def get_moves_to_target(start_pos, target_pos, start_angle, perfect_angle=False):
    '''
    Get the direction and turns needed to head towards a certain direction
    
    perfect_angle: if True, assume we can reach the perfect angle needed rather
        than requiring harder turns
    '''
    target_vector = target_pos - start_pos
    target_angle = np.arctan2(target_vector[1], target_vector[0])

    # Calculate the angle needed to head in vector direction to target
    #  and how many turns are needed to get there
    # theta = abs(target_angle - angle) % (2 * np.pi)
    # min_angle_dist = 2 * np.pi - theta if theta > np.pi else theta
    # direction = -1 if theta > np.pi else 1
    
    angle_dist1 = target_angle - start_angle
    angle_dist2 = min(abs(2*np.pi+angle_dist1), abs(2*np.pi-angle_dist1))
    sign1 = np.sign(angle_dist1)
    angle_dist1 = abs(angle_dist1)

    if angle_dist1 < angle_dist2:
        direction = sign1
    else:
        direction = -1 * sign1
    min_angle_dist = min(angle_dist1, angle_dist2)
    
    turns = np.round(min_angle_dist / turn_speed)
    
    if perfect_angle:
        resulting_angle = target_angle
    else:
        resulting_angle = (direction * turns * turn_speed + start_angle) % (2*np.pi)
    resulting_vector = np.array([np.cos(resulting_angle), np.sin(resulting_angle)])

    return resulting_angle, resulting_vector, turns


def compute_path_steps(pos, angle, target, ret_pos=False, perfect_angle=False, 
                      ret_steps=False):
    '''Compute a theoretical best bath through the given target
    
    ret_pos: whether to return the intermediate positions
    perfect_angle: whether to assume that we can turn perfectly to the correct angle
    ret_steps: whether to return the individual steps used'''
    
    # Find number of steps to get above y of 250 via shortcut
    angle1, dir1, turns1 = get_moves_to_target(pos, target, angle, perfect_angle)
    steps1 = np.ceil((250 - pos[1]) / (move_speed * dir1[1]))
    steps1 = steps1//2
    pos1 = steps1 * move_speed * dir1 + pos

    angle1b, dir1b, turns1b = get_moves_to_target(pos1, target, angle1, perfect_angle)
    steps1b = np.ceil((250 - pos1[1]) / (move_speed * dir1b[1]))
    pos1b = steps1b * move_speed * dir1b + pos1


    angle2, dir2, turns2 = get_moves_to_target(pos1b, goal_target, angle1b, perfect_angle)
    steps2 = np.ceil((262.5 - pos1b[0]) / (move_speed * dir2[0]))
    steps2 = steps2//2
    pos2 = steps2 * move_speed * dir2 + pos1b

    angle2b, dir2b, turns2b = get_moves_to_target(pos2, goal_target, angle2, perfect_angle)
    steps2b = np.ceil((262.5 - pos2[0]) / (move_speed * dir2b[0]))
    pos2b = steps2b * move_speed * dir2b + pos2

    total_steps = turns1 + turns1b + turns2 + turns2b + steps1 + steps1b + steps2 + steps2b
    
    if ret_pos:
        if ret_steps:
            return total_steps, [pos1, pos1b, pos2, pos2b], [turns1, turns1b, turns2, turns2b, steps1, steps1b, steps2, steps2b]
        return total_steps, [pos1, pos1b, pos2, pos2b]
    elif ret_steps:
        return total_steps, [turns1, turns1b, turns2, turns2b, steps1, steps1b, steps2, steps2b]
    
    return total_steps


def check_shortcut_usage(p, ret_arrays=False):
    '''
    Check if shortcut was used in a trajectory
    '''
    # Check that x values are within range
    x1 = (p[:-1, 0] > 125) & (p[:-1, 0] < 175)
    x2 = (p[1:, 0] > 125) & (p[1:, 0] < 175)

    # Check that y values cross over
    y1 = (p[:-1, 1] < 250)
    y2 = (p[1:, 1] > 250)

    # Crossed shortcut
    used_shortcut = ((y1 & y2) & (x1 | x2)).any()
    
    if ret_arrays:
        return used_shortcut, [x1, x2, y1, y2]
    
    return used_shortcut



def test_shortcut_agent(model, obs_rms, character_reset_pos=0, n_eps=20):
    env_kwargs = {'character_reset_pos': character_reset_pos, 'shortcut_probability': 1.}
    res1 = evaluate(model, obs_rms, env_kwargs=env_kwargs, env_name='ShortcutNav-v0',
                  num_episodes=n_eps, data_callback=shortcut_data_callback)

    env_kwargs = {'character_reset_pos': character_reset_pos, 'shortcut_probability': 0}
    res2 = evaluate(model, obs_rms, env_kwargs=env_kwargs, env_name='ShortcutNav-v0',
                  num_episodes=n_eps, data_callback=shortcut_data_callback)

    # Check how often shortcuts were used when available
    shortcuts_used = np.array([check_shortcut_usage(p) for p in res1['data']['pos']])

    # Check how well agent did compared to an "optimal" path
    shortcut_starts = []
    shortcut_optimal_steps = []
    for i in range(n_eps):
        # First position and angle of the episode
        p = res1['data']['pos'][i][0]
        a = res1['data']['angle'][i][0]
        shortcut_starts.append((p, a))
        shortcut_optimal_steps.append(compute_path_steps(p, a, shortcut_target, perfect_angle=True))

    path_optimal_steps = []
    path_starts = []
    for i in range(n_eps):
        p = res2['data']['pos'][i][0]
        a = res2['data']['angle'][i][0]
        path_starts.append((p, a))
        path_optimal_steps.append(compute_path_steps(p, a, path_target, perfect_angle=True))

    shortcut_actual_steps = [len(r) for r in res1['rewards']]
    path_actual_steps = [len(r) for r in res2['rewards']]
    
    return {
        'shortcuts_used': shortcuts_used,
        'shortcut_use_rate': np.sum(shortcuts_used)/n_eps,
        'shortcut_actual_steps': shortcut_actual_steps,
        'shortcut_theoretical_steps': np.array(shortcut_optimal_steps).squeeze() - 2,
        'shortcut_starts': shortcut_starts,
        'path_actual_steps': path_actual_steps,
        'path_theoretical_steps': np.array(path_optimal_steps).squeeze() - 2,
        'path_starts': path_starts
    }


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def shortcut_color_pos(pos):
    rectangles = [ #each rectangle given by x1, x2, y1, y2
        [0, 125, 250, 300], #red
        [125, 300, 250, 300], #green
        [0, 300, 0, 250], #blue

        [0, 100, 150, 250], #purple
        [100, 200, 150, 250], #cyan
    ]

    colors = [
        [0.9, 0., 0., 0.3],
        [0., 0.9, 0., 0.3],
        [0., 0., 0.9, 0.3],

        [0.9, 0., 0.9, 0.3],
        [0., 0.9, 0.9, 0.3]
    ]
    
    rect_idxs = []
    
    for rectangle in rectangles:
        x1, x2, y1, y2 = rectangle
        idxs = (pos[:, 0] >= x1) & (pos[:, 0] <= x2) & (pos[:, 1] >= y1) & (pos[:, 1] <= y2)
        rect_idxs.append(idxs)

    return {
        'rect_idxs': rect_idxs,
        'colors': colors
    }

def test_shortcut_activations(exp_string='shortcutnav_fcp{prob}reset{reset}batch{batch}',
                              prob=0.1, reset=3, batch=64, trial=0, chk=300,
                              umap_min_dist=0.5, skip_activs=True, skip_umap=True):
    exp_name = exp_string.format(prob=prob, reset=reset, batch=batch)
    model, obs_rms = load_chk(exp_name, chk, trial) #note that this load has a subfolder baked in
    
    
    shortcut_res = test_shortcut_agent(model, obs_rms, reset)
    
    env_kwargs = {'character_reset_pos': reset, 'shortcut_probability': 0.5}
    res = evaluate(model, obs_rms, env_name='ShortcutNav-v0', env_kwargs=env_kwargs,
                   data_callback=shortcut_data_callback, with_activations=True,
                   num_episodes=100)

    # Process activations
    activs = ep_stack_activations(res, True)
    activ = activs['shared_activations'][1]
    ep_pos = res['data']['pos']
    pos = np.vstack(ep_pos)
    angle = np.vstack(res['data']['angle']).reshape(-1)
    # activ = torch.hstack([a['shared_activations'] for a in activs])[1]
    umap = UMAP(min_dist=umap_min_dist)
    activ2d = umap.fit_transform(activ)
    
    xlim1, ylim1 = activ2d.min(axis=0)
    xlim2, ylim2 = activ2d.max(axis=0)
    norm_activ2d = activ2d.copy()
    norm_activ2d[:, 0] = (norm_activ2d[:, 0] - xlim1) / (xlim2 - xlim1)
    norm_activ2d[:, 1] = (norm_activ2d[:, 1] - ylim1) / (ylim2 - ylim1)

    # Process shortcut availability and whether it was used
    ep_shortcut_avail = []
    ep_shortcut_used = []
    for i in range(len(activs)):
        num_steps = activs[i]['shared_activations'].shape[1]
        ep_shortcut_avail.append(np.full((num_steps, 1), res['data']['shortcut'][i][0]))
        shortcut_used = check_shortcut_usage(ep_pos[i])
        ep_shortcut_used.append(np.full((num_steps, 1), shortcut_used))
    shortcut_avail = np.vstack(ep_shortcut_avail).reshape(-1)
    shortcut_used = np.vstack(ep_shortcut_used).reshape(-1)
    
    chunks = shortcut_color_pos(pos)
    
    data = {
        'ep_pos': res['data']['pos'],
        'pos': pos,
        'angle': angle,
        'actions': np.vstack(res['actions']).reshape(-1),
        'activ2d': activ2d,
        'shortcut_avail': shortcut_avail,
        'shortcut_used': shortcut_used,
        'pos_chunks': chunks,
        'shortcut_use_rate': shortcut_res['shortcut_use_rate']
    }
    if not skip_activs:
        activs = ep_stack_activations(res, True)
        data['activs'] = activs
    if not skip_umap:
        data['umap'] = umap
    return data


def test_shortcut_activations_chks(exp_string='shortcutnav_fcp{prob}reset{reset}batch{batch}',
                              prob=0.1, reset=3, batch=64, trial=0, chks=[40, 70, 100, 150, 200, 300, 400],
                              verbose=1, umap_min_dist=0.5):
    all_shortcut_res = []
    
    if verbose > 0:
        chks = tqdm(chks)
    
    for chk in chks:
        shortcut_res = test_shortcut_activations(exp_string=exp_string, 
                                                 prob=prob, reset=reset, batch=batch,
                                                 trial=trial, chk=chk, umap_min_dist=umap_min_dist)
        all_shortcut_res.append(shortcut_res)
    return all_shortcut_res


def plot_shortcut_umap(shortcut_res, plot_type=0, ax=None, normalize=True):
    '''
    Plot some umap heatmaps
    plot_type:
        -1: Plain umap no coloring
        0: Color based on whether shortcut was available
        1: Color based on what position chunk was it
        2: Color based on position chunk and plot onto multiple axes (6 total)
        3: Color based on whether shortcut was used for trajectory
    '''
    if ax is None and plot_type in [-1, 0, 1, 3]:
        fig, ax = pplt.subplots()
    if ax is None and plot_type in [2]:
        fig, ax = pplt.subplots(ncols=6)
    
    activ2d = shortcut_res['activ2d'].copy()
    if normalize:
        activ2d = normalize_umap(activ2d)
        
    if plot_type == -1:
        ax.scatter(activ2d.T[0], activ2d.T[1], alpha=0.3)
    elif plot_type == 0:
        shortcut_avail = shortcut_res['shortcut_avail']
        ax.scatter(activ2d.T[0], activ2d.T[1], c=shortcut_avail*2-1, alpha=0.3)
    elif plot_type == 1:
        colors = shortcut_res['pos_chunks']
        for i, idxs in enumerate(colors['rect_idxs']):
            color = colors['colors'][i]
            a = activ2d[idxs]
            ax.scatter(a.T[0], a.T[1], color=color, alpha=0.3)
        
    elif plot_type == 2:
        colors = shortcut_res['pos_chunks']
        for i, idxs in enumerate(colors['rect_idxs']):
            color = colors['colors'][i]
            a = activ2d[idxs]
            ax[0].scatter(a.T[0], a.T[1], color=color, alpha=0.3)
            ax[i+1].scatter(a.T[0], a.T[1], color=color, alpha=0.3)
            
    elif plot_type == 3:
        ur_map_idxs = shortcut_res['pos_chunks']['rect_idxs'][1]
        shortcut_used = shortcut_res['shortcut_used']

        a1 = activ2d[ur_map_idxs & shortcut_used]
        ax.scatter(a1.T[0], a1.T[1], color='green', alpha=0.3)
        a2 = activ2d[ur_map_idxs & ~shortcut_used]
        ax.scatter(a2.T[0], a2.T[1], color='red', alpha=0.3)

        X = np.vstack([a1, a2])
        labels = [0]*len(a1) + [1]*len(a2)

        if len(a1) > 0 and len(a2) > 0:
            score = silhouette_score(X, labels)
        else:
            score = 0
            
        return score
    
    
def normalize_umap(um2d):
    xlim1, ylim1 = um2d.min(axis=0)
    xlim2, ylim2 = um2d.max(axis=0)
    um2d[:, 0] = (um2d[:, 0] - xlim1) / (xlim2 - xlim1)
    um2d[:, 1] = (um2d[:, 1] - ylim1) / (ylim2 - ylim1)
    return um2d

    
def compute_shortcut_silhouette(shortcut_res):
    activ2d = shortcut_res['activ2d'].copy()
    ur_map_idxs = shortcut_res['pos_chunks']['rect_idxs'][1]
    shortcut_used = shortcut_res['shortcut_used']

    a1 = activ2d[ur_map_idxs & shortcut_used]
    a2 = activ2d[ur_map_idxs & ~shortcut_used]
    X = np.vstack([a1, a2])
    labels = [0]*len(a1) + [1]*len(a2)

    if len(a1) > 0 and len(a2) > 0:
        score = silhouette_score(X, labels)
    else:
        score = 0
        
    return score



n_clusters = 5
cluster_labels = ['Start', 'Entrance', 'Shortcut', 'Corridor Entrance', 'Near Platform']

def get_ep_lens(ep_pos):
    ep_lens = [len(ep) for ep in ep_pos]
    return ep_lens

def ep_split_res(target, ep_lens):
    data = []
    cur_idx = 0
    for l in ep_lens:
        data.append(target[cur_idx:cur_idx+l])
        cur_idx += l
    return data


def kmeans_activ_pos_plot(pos, activ2d, labels=None, ax=None, n_clusters=5, km=None):
    '''
    Plot kmeans activ2d plot with positions
    Can give a combination of
    activ2d: perform kmeans on this
    activ2d/km: perform kmeans using an already fit km object
    labels: use existing labels
    '''
    format_ax = False
    if ax is None:
        format_ax = True
        fig, ax = pplt.subplots(ncols=n_clusters+1, sharey=False, sharex=False)
        
    if labels is None:
        if km is None:
            km = KMeans(n_clusters=n_clusters)
            labels = km.fit_predict(activ2d)
        else:
            labels = km.predict(activ2d)

    cluster_range = km.n_clusters if km is not None else n_clusters
    for i in range(cluster_range):
        idxs = labels == i
        ax[0].scatter(activ2d[idxs].T[0], activ2d[idxs].T[1], color=rgb_colors[i], alpha=0.1)
        ax[i+1].scatter(pos[idxs].T[0], pos[idxs].T[1], color=rgb_colors[i], alpha=0.1)
        
    if format_ax:
        ax[1:].format(xlim=[0, 300], ylim=[0, 300])
        
        
        
def shortcut_km_cluster_prob2(ep_labels, pos):
    cluster_names = ['start', 'entry', 'shortcut', 'corridor', 'platform']
    cluster_idxs = [] #which label is each cluster most likely to be
    cluster_probs = []
    n_clusters=5
    labels = np.concatenate(ep_labels)

    no_remap_conflict = True
    first_labels = np.array([l[0] for l in ep_labels])
    for i in range(n_clusters):
        # are positions mostly above or below corridor?
        p = pos[labels == i]
        above_corridor = (p[:, 1] > 250).sum() / len(p)
        left = (p[:, 0] < 175).sum() / len(p)

        p2 = p[(p[:, 1] > 240) & (p[:, 1] < 260)]
        if len(p2) > 4:
            left2 = (p2[:, 0] < 125).sum() / len(p2)
            right2 = 1 - left2
        else:
            left2 = 0.
            right2 = 0.
        start = (first_labels == i).sum() / len(first_labels)

        prios = []
        if left > 0.5: #mostly left corridor
            above_prios = [3, 4]
        else:
            above_prios = [4, 3]

        if left2 > 0.5: #mostly left entrance
            below_prios = [1, 2]
        else:
            below_prios = [2, 1]

        if start > 0.5:
            prios += [0]
            if above_corridor > 0.5:
                prios += above_prios
                prios += below_prios
            else:
                prios += below_prios
                prios += above_prios
        else:
            if above_corridor > 0.5:
                prios += above_prios
                prios += below_prios
            else:
                prios += below_prios
                prios += above_prios
            prios += [0]

        for j, prio in enumerate(prios):
            if prio not in cluster_idxs:
                cluster_idxs.append(prio)
                break
            no_remap_conflict = False
                
        probs = [start, (1-above_corridor)*left2, (1-above_corridor)*right2,
                              above_corridor*left, above_corridor*(1-left)]
        cluster_probs.append(probs)
    
    # Note this is indexed as cluster_probs[cluster_idx][cluster_label likelihood]
    cluster_probs = np.vstack(cluster_probs)
    
    relabel_map = {i: cluster_idxs[i] for i in range(n_clusters)}
    
    reorder_idxs = np.argsort(list(relabel_map.values()))
    keys = list(relabel_map.keys())
    values = list(relabel_map.values())
    relabel_map2 = {}
    for idx in reorder_idxs:
        relabel_map2[values[idx]] = keys[idx]
    reordered_cluster_probs = cluster_probs[list(relabel_map2.values()), :]
    
    # Calculate transition probabilities while we are here
    reordered_ep_labels = [relabel_cluster(e, relabel_map) for e in ep_labels]
    transition_probs = shortcut_km_transition_probs(reordered_ep_labels)

    return {
        'cluster_probs': cluster_probs, 
        'relabel_map': relabel_map, 
        'cluster_names': cluster_names, 
        'no_remap_conflict': no_remap_conflict,
        'reordered_cluster_probs': reordered_cluster_probs,
        'reordered_transition_probs': transition_probs
    }


def relabel_cluster(labels, relabel_map):
    '''
    Relabel a batch of labels to fit the relabelled schema
    '''
    relabelled = labels.copy()
    for i in range(len(labels)):
        relabelled[i] = relabel_map[labels[i]]
    return relabelled


def shortcut_km_transition_probs(ep_labels, n_clusters=5, as_probs=True):
    '''
    Given episodic labels, track how likely transitions are from one cluster to the next
    
    as_probs: compute transition probabilities instead of counts
    '''
    transitions = np.zeros((n_clusters, n_clusters))
    for l in ep_labels:
        for i in range(len(l)-1):
            start = l[i]
            end = l[i+1]
            transitions[start][end] += 1
            
    if as_probs:
        for i in range(n_clusters):
            transitions[i] = transitions[i] / transitions[i].sum()
            
    return transitions



def shortcut_cluster_analysis(s, normalize=True):
    '''Pass a shortcut_res object to perform clustering analysis'''
    n_clusters = 5
    activ2d = s['activ2d'].copy()
    if normalize:
        xmin, ymin = activ2d.min(axis=0)
        xmax, ymax = activ2d.max(axis=0)
        activ2d[:, 0] = (activ2d[:, 0] - xmin) / (xmax - xmin)
        activ2d[:, 1] = (activ2d[:, 1] - ymin) / (ymax - ymin)
    
    pos = s['pos']
    ep_lens = get_ep_lens(s['ep_pos'])
    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(activ2d)
    ep_labels = ep_split_res(labels, ep_lens)

    res = shortcut_km_cluster_prob2(ep_labels, pos)
    relabel_map = res['relabel_map']
    
    labels = relabel_cluster(labels, relabel_map)
    
    relabelled_centers = np.zeros(km.cluster_centers_.shape)
    for i in range(n_clusters):
        relabelled_centers[relabel_map[i]] = km.cluster_centers_[i]
        
    res['labels'] = labels
    res['cluster_centers'] = relabelled_centers
    
    t = res['reordered_transition_probs'].copy()
    for i in range(len(t)):
        t[i, i] = 0
        if t[i].sum() != 0:
            t[i] = t[i] / t[i].sum()
        else:
            t[i] = 0.
    res['t_prob'] = t
    res['km'] = km
    
    res['ep_activ2d'] = ep_split_res(activ2d, ep_lens)
    res['activ2d'] = activ2d
    return res


def linear_bestfit(x, y):
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)
    
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    lr = LinearRegression().fit(x, y)
    y_pred = lr.predict(x)
    
    x_min, x_max = x.min(), x.max()
    r2 = r2_score(y, y_pred)
    
    xs = np.array([x_min, x_max]).reshape(-1, 1)
    ys = lr.predict(xs)
    
    return xs, ys, r2

