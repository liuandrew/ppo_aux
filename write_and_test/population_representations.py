
import sys
sys.path.append('../')

import numpy as np
from shortcut_analysis import get_ep_lens, check_shortcut_usage
from umap import UMAP
import torch
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter


wsns_folders = {
    'control': 'wc1.5_policy',
    'plum': 'plum2sc_policy',
    'aux': 'sc_aux_policy',
    'sc2sc': 'sc2sc_policy'
}

wsns_formats = {
    'control': '{p}_{t}',
    'plum': '{p}_{task}_{t}',
    'aux': '{p}_{aux}_{t}',
    'sc2sc': '{p}_{clone}_{t}',
}

summaries = {
    'control': ['summary'],
    'plum': ['0.1_summary', '0.2_summary'],
    'aux': ['0.1_summary', '0.2_summary'],
    'sc2sc': ['0.1_summary', '0.2_summary'],
}

summary_iterators = {
    'control': [0.1, 0.2, 0.3, 0.4, 0.5, 0.8],
    'plum': ['1.7', '2.7', '3tp0.1', '3tp0.4'],
    'aux': ['catquad', 'catwall01', 'wall01', 'catfacewall', 'catshort'],
    # 'sc2sc': ['shared', 'actor1'],
    'sc2sc': ['shared'],
}

summary_labels = {
    'control': [f'p={p}' for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.8]],
    'plum': ['Grid', 'Maze', 'p=0.1 Plum', 'p=0.4 Plum'],
    'aux': ['Quad.', 'Cat. N/E', 'Num. N/E', 'Faced Wall', 'Short. Avail'],
    'sc2sc': ['Clone Shared', 'Clone Actor']
}

exp_labels = ['Control', 'Auxiliary Tasks', 'Plum Transfer', 'Transfer']

all_labels = {
    0.1: 'p=0.1', 0.2: 'p=0.2', 0.3: 'p=0.3', 0.4: 'p=0.4', 0.5: 'p=0.5', 0.8: 'p=0.8',
    
    '0.1_catquad': 'Quad.', '0.2_catquad': 'Quad.',
    '0.1_catwall01': 'Cat. N/E', '0.2_catwall01': 'Cat. N/E',
    '0.1_wall01': 'Num. N/E', '0.2_wall01': 'Num. N/E',
    '0.1_catfacewall': 'Faced Wall', '0.2_catfacewall': 'Faced Wall',
    '0.1_catshort': 'Short. Avail', '0.2_catshort': 'Short. Avail',
    
    '0.1_1.7': 'Grid', '0.2_1.7': 'Grid',
    '0.1_2.7': 'Maze', '0.2_2.7': 'Maze',
    '0.1_3tp0.1': 'p=0.1 Plum', '0.2_3tp0.1': 'p=0.1 Plum',
    '0.1_3tp0.4': 'p=0.4 Plum', '0.2_3tp0.4': 'p=0.4 Plum',
    
    # '0.1_shared': 'Clone Shared', '0.2_shared': 'Clone Shared',
    '0.1_shared': 'Cloned from p=0.5', '0.2_shared': 'Cloned from p=0.5',
    '0.1_actor1': 'Clone Actor', '0.2_actor1': 'Clone Actor',
}

# Used to only plot p=0.1 or p=0.2 and not modify plotting code too much
summary_p = {
    0.1: {
        'summaries': {
            'control': ['summary'], 'plum': ['0.1_summary'],
            'aux': ['0.1_summary'], 'sc2sc': ['0.1_summary'],
        },
        'iterators': {
            'control': [0.1], 'plum': ['1.7', '2.7', '3tp0.1', '3tp0.4'],
            'aux': ['catquad', 'catwall01', 'wall01', 'catfacewall', 'catshort'],
            'sc2sc': ['shared', 'actor1']
        },
        'labels': {
            'control': ['p=0.1'], 'plum': ['Grid', 'Maze', 'p=0.1 Plum', 'p=0.4 Plum'],
            'aux': ['Quad.', 'Cat. N/E', 'Num. N/E', 'Faced Wall', 'Short. Avail'],
            'sc2sc': ['Clone Shared', 'Clone Actor']
        }
    },
    0.2: {
        'summaries': {
            'control': ['summary'], 'plum': ['0.2_summary'],
            'aux': ['0.2_summary'], 'sc2sc': ['0.2_summary'],
        },
        'iterators': {
            'control': [0.2], 'plum': ['1.7', '2.7', '3tp0.1', '3tp0.4'],
            'aux': ['catquad', 'catwall01', 'wall01', 'catfacewall', 'catshort'],
            'sc2sc': ['shared', 'actor1']
        },
        'labels': {
            'control': ['p=0.2'], 'plum': ['Grid', 'Maze', 'p=0.1 Plum', 'p=0.4 Plum'],
            'aux': ['Quad.', 'Cat. N/E', 'Num. N/E', 'Faced Wall', 'Short. Avail'],
            'sc2sc': ['Clone Shared', 'Clone Actor']
        }        
    }
}

# Individual experiment locations

exp_folders = {
    'control': 'shortcut_wc2',
    'plum': 'plumtosc',
    'aux': 'shortcut_aux',
    'sc2sc': 'sctosc',
}

exp_formats = {
    'control': 'shortcut_wc1.5p{p}_t{t}',
    'plum': 'plumtosc_shared_task{task}p{p}_t{t}',
    'aux': 'shortcut_wc1.5p{p}_aux{aux}_t{t}',
    'sc2sc': 'sctosc_{clone}_p0.5top{p}_t{t}',
}

cont_formats = {
    'control': 'shortcut_wc1.5p{p}_cont_t{t}',
    'plum': 'plumtosc_shared_task{task}p{p}_cont_t{t}',
    'aux': 'shortcut_wc1.5p{p}_aux{aux}_cont_t{t}',
    'sc2sc': 'sctosc_{clone}_p0.5top{p}_cont_t{t}',
}

exp_iterators = {
    'control': {
        'p': [0.1, 0.2, 0.3, 0.4, 0.5, 0.8]
    },
    'plum': {
        'p': [0.1, 0.2],
        'task': ['1.7', '2.7', '3tp0.1', '3tp0.4']
    },
    'aux': {
        'aux': ['catquad', 'catwall01', 'wall01', 'catfacewall', 'catshort'],
        'p': [0.1, 0.2]
    },
    'sc2sc': {
        'clone': ['shared', 'actor1'],
        'p': [0.1, 0.2]
    }
}

def get_exp_keys(exp, p=None):
    '''
    Get the iterators needed to go through an intervention experiment set
    exp: type of experiment: 'control'/'plum'/'aux'/'sc2sc'
    p: specficy which p, e.g. 0.1/0.2, for interventions. If None, will give
        an iterator for both
    '''
    if exp == 'control':
        if p == 0.1: return [0.1]
        if p == 0.2: return [0.2]
        return summary_iterators[exp]

    it = summary_iterators[exp]    
    if p is None:
        its = []
        for p in [0.1, 0.2]:
            for i in it:
                its.append(f'{p}_{i}')
        return its

    if p not in [0.1, 0.2]:
        raise Exception('p should be one of 0.1 or 0.2 if passed')
    
    its = []
    for i in it:
        its.append(f'{p}_{i}')
    return its    




'''

General helper functions for dealing with WS/NS (with shortcut, no shortcut)
experiment run files. These were generated primarily in shortcut_experiments.ipynb.

'''

def combine_wsns(wsns_res, chk=0, use_real_chk=False):
    '''
    Combine the ws/ns episodes from our 100 episode trial experiments
    chk: the index of chk from chk_sched. 
        For example, chk=25 will be checkpoint 420 or 2.7e6 timesteps after
        shifted start of training
        
    use_real_chk: if True, assume the chk passes an actual checkpoint rather than index of chk_sched
    '''
    
    if not use_real_chk:
        chks = list(wsns_res['ws'].keys())
        if chk >= len(chks):
            return False
        chk = chks[chk]
    
    ws_res, ns_res = wsns_res['ws'][chk], wsns_res['ns'][chk]
    ws_activ = [a['shared_activations'][1] for a in ws_res['activs']]
    ns_activ = [a['shared_activations'][1] for a in ns_res['activs']]
    ws_pos, ns_pos = ws_res['ep_pos'], ns_res['ep_pos']
    ws_angle, ns_angle = ws_res['ep_angle'], ns_res['ep_angle']
    ws_lens, ns_lens = get_ep_lens(ws_pos), get_ep_lens(ns_pos)
    ws_used, ns_used = [check_shortcut_usage(p) for p in ws_res['ep_pos']], [False]*50
    ws_avail, ns_avail = ws_res['shortcut'], ns_res['shortcut']
    ws_vis, ns_vis = ws_res['vis'], ns_res['vis']
    
    activ2d = None
    if 'activ2d' in wsns_res and chk in wsns_res['activ2d']:
        activ2d = wsns_res['activ2d'][chk]

    activ = torch.vstack(ws_activ+ns_activ)
    ep_pos = ws_pos + ns_pos
    pos = np.vstack(ep_pos)
    ep_angle = ws_angle + ns_angle
    angle = np.concatenate(ep_angle)
    ep_vis = ws_vis + ns_vis
    vis = np.concatenate(ep_vis)
    used = ws_used + ns_used
    available = ws_avail + ns_avail

    ws_lens, ns_lens = get_ep_lens(ws_pos), get_ep_lens(ns_pos)
    lens = ws_lens + ns_lens

    return {
        'ep_activ': ws_activ+ns_activ,
        'activ': activ,
        'ep_pos': ep_pos,
        'pos': pos,
        'ep_angle': ep_angle,
        'angle': angle,
        'used': used,
        'available': available,
        'ep_vis': ep_vis,
        'vis': vis,
        'lens': lens,
        'chk': chk,
        'activ2d': activ2d
    }



def add_activ2d_to_wsns(wsns_res, chk=0, use_real_chk=False):
    '''
    Add activ2d to a ws/ns episodes save
    '''
    comb = combine_wsns(wsns_res, chk, use_real_chk)
    if not comb:
        return None
    
    activ = comb['activ']
    lens = comb['lens']
    save_chk = comb['chk']
    
    umap = UMAP(min_dist=0.35)
    activ2d = umap.fit_transform(np.vstack(activ)).astype('float16')
    
    ep_activ2d = ep_split_res(activ2d, lens)
    ws_activ2d, ns_activ2d = ep_activ2d[:50], ep_activ2d[50:]
    
    if 'activ2d' not in wsns_res:
        wsns_res['activ2d'] = {}
        
    wsns_res['activ2d'][save_chk] = activ2d
    wsns_res['ws'][save_chk]['activ2d'] = ws_activ2d
    wsns_res['ns'][save_chk]['activ2d'] = ns_activ2d
    
    return True


'''

Population representation analysis functions

'''



def point_cloud_heatmap(points, all_data=None, resolution=50, sigma=1):
    '''Turn a 2D point cloud data into heatmap'''
    if all_data is not None:
        a_range = np.vstack([all_data.min(axis=0), all_data.max(axis=0)]).T
    else:
        all_data = points
        a_range = np.vstack([all_data.min(axis=0), all_data.max(axis=0)]).T
    
    a_extent = [a_range[0, 0], a_range[0, 1], a_range[1, 0], a_range[1, 1]]
    heatmap = np.histogram2d(points[:, 0], points[:, 1], bins=50, range=a_range)[0]
    
    # low-pass
    heatmap = low_pass_heatmap_filter(heatmap)
    # smooth
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    # flatten
    heatmap = low_pass_heatmap_filter(heatmap, flatten=True)
    
    return heatmap, a_range, a_extent
    
    
def low_pass_heatmap_filter(heatmap, bin_num=1, flatten=False):
    '''Perform a low-papss filter on heatmap flexibly based on histogram bins
    
    bin_num: which bin to filter by from histogram
    flatten: if True, set the value of any points that survive the filter to 1
    '''
    bins = np.histogram(heatmap[heatmap > 0].flatten())[1]
    heatmap[heatmap < bins[bin_num]/2] = 0
    if flatten:
        heatmap[heatmap >= bins[bin_num]/2] = 1
    return heatmap



def dist_to_hm(points, hm, extent, resolution=50):
    '''
    Calculate the minimum distances between each point in points to the closest part of a heatmap
    hm: the heatmap, potentially generated by point_cloud_heatmap
    extent: the extent of values covered by the heatmap, also from point_cloud_heatmap
    '''
    x_grid = np.linspace(extent[0], extent[1], resolution)
    y_grid = np.linspace(extent[2], extent[3], resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    X, Y = X.T, Y.T    
    xs, ys = X[hm == 1], Y[hm == 1] 
    hm_points = np.vstack([xs, ys]).T # points representing the heatmap

    dists = distance.cdist(points, hm_points)
    dists = dists.min(axis=1)
    
    return dists


def hm_overlap(hm1, hm2):
    non_overlap1 = np.where((hm1 != 0) & (hm2 == 0), hm1, 0)
    non_overlap2 = np.where((hm2 != 0) & (hm1 == 0), hm2, 0)
    non_overlap = non_overlap1 + non_overlap2

    overlap = np.minimum(hm1, hm2)
    
    return overlap, non_overlap1, non_overlap2, non_overlap


def get_hm_and_overlap(comb, label1=None, label2=None, max_point_dist=0.5):
    '''Take a comb object (from combine_wsns, or otherwise should have data of
    used, ep_pos, pos, activ2d) and process with the following
    
    1. Get point labels based on shortcut trajectory
    2. Turn each label into heatmaps
        - Add these labels and heatmaps to the comb object
    3. If label0, label1 given, find overlap and unique parts of
        each heatmap associated
    4. Find the points associated with overlap and unique parts
    
    label1, label2: the labels from decompose_shortcut_trajectories to consider heatmap overlaps for
        E.g., 0: orange, 1: green, 2: red, 3: purple
    max_point_dist: activation distance to consider part of a heatmap or not
    '''
    used = comb['used']
    ep_pos = comb['ep_pos']
    pos = comb['pos']
    activ2d = comb['activ2d']
    labels, _ = decompose_shortcut_trajectories(ep_pos=ep_pos, ep_shortcut_used=used)

    hms = {}
    # for label in np.unique(labels):
    for label in range(-1, 4):
        a = activ2d[labels == label]
        hm, a_range, a_extent = point_cloud_heatmap(a, activ2d)
        hms[label] = hm

    comb['labels'] = labels
    comb['hms'] = hms
    comb['a_extent'] = a_extent
    comb['a_range'] = a_range
        
    if label1 is not None and label2 is not None:
        hm1, hm2 = hms[label1], hms[label2]
        overlap, non_overlap1, non_overlap2, non_overlap = hm_overlap(hm1, hm2)
        
        results = {
            'names': ['overlap', 'non_overlap1', 'non_overlap2', 'non_overlap'],
            'hms': [overlap, non_overlap1, non_overlap2, non_overlap],
            'point_idxs': [], # what points correspond to belonging to an overlap heatmap
        }
        
        for hm in [overlap, non_overlap1, non_overlap2, non_overlap]:
            if hm.sum() == 0:
                results['point_idxs'].append(np.full(len(activ2d), False))
            else:
                dists = dist_to_hm(activ2d, hm, a_extent)
                idxs = dists < max_point_dist
                results['point_idxs'].append(idxs)
        
        return results
            
    
def point_overlap_portion(comb, label0, label1):
    '''Compute the portion of points that belong to the overlap of a cluster
    
    comb: combined WS/NS results from combine_wsns
    label0, label1: clusters to check overlap score for
    
    returns: portion of points overlapped in cluster0 and cluster1
    '''
    results = get_hm_and_overlap(comb, label0, label1)

    l0, l1 = comb['labels'] == label0, comb['labels'] == label1
    a0, a1 = comb['activ2d'][l0], comb['activ2d'][l1]
        
    if (l0.sum() == 0) or (l1.sum() == 0):
        return 0., 0. 
    
    pts0 = dist_to_hm(a0, comb['hms'][label0], comb['a_extent'])
    pts1 = dist_to_hm(a1, comb['hms'][label1], comb['a_extent'])
    pts0, pts1 = (pts0 < 0.5).sum(), (pts1 < 0.5).sum()

    # compute portion that is overlapped
    idx = results['point_idxs'][0]
    
    if pts0 == 0:
        overlap0 = 0
    else:
        overlap0 = ((comb['labels'] == label0) & idx).sum() / pts0

    if pts1 == 0:
        overlap1 = 0
    else:
        overlap1 = ((comb['labels'] == label1) & idx).sum() / pts1

    return overlap0, overlap1
    