import gym
import torch
import gym_nav
import numpy as np

import sys
sys.path.append('../')
from evaluation import *
from shortcut_analysis import *
from explore_analysis import *

from tqdm import tqdm
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import time

from pathlib import Path

pplt.rc.update({'font.size': 10})


def combine_wsns(wsns_res, chk=0):
    '''
    Combine the ws/ns episodes from our 100 episode trial experiments
    chk: the index of chk from chk_sched. 
        For example, chk=25 will be checkpoint 420 or 2.7e6 timesteps after
        shifted start of training
    '''
    chks = list(wsns_res['ws'].keys())
    if chk >= len(chks):
        return False

    chk = chks[chk]
    ws_res, ns_res = wsns_res['ws'][chk], wsns_res['ns'][chk]
    ws_activ = [a['shared_activations'][1] for a in ws_res['activs']]
    ns_activ = [a['shared_activations'][1] for a in ns_res['activs']]
    ws_pos, ns_pos = ws_res['ep_pos'], ns_res['ep_pos']
    ws_lens, ns_lens = get_ep_lens(ws_pos), get_ep_lens(ns_pos)
    ws_used, ns_used = [check_shortcut_usage(p) for p in ws_res['ep_pos']], [False]*50

    activ = torch.vstack(ws_activ+ns_activ)
    ep_pos = ws_pos + ns_pos
    used = ws_used + ns_used
    pos = np.vstack(ep_pos)

    ws_lens, ns_lens = get_ep_lens(ws_pos), get_ep_lens(ns_pos)
    lens = ws_lens + ns_lens

    return {
        'ep_activ': ws_activ+ns_activ,
        'activ': activ,
        'ep_pos': ep_pos,
        'pos': pos,
        'used': used,
        'lens': lens,
        'chk': chk
    }



def collect_activ2d_from_wsns(wsns_res, activ2d_data, chk=0):
    '''
    Add activ2d to a ws/ns episodes save
    '''
    comb = combine_wsns(wsns_res, chk)
    if not comb:
        return None
    
    activ = comb['activ']
    lens = comb['lens']
    save_chk = comb['chk']
    
    umap = UMAP(min_dist=0.35)
    activ2d = umap.fit_transform(np.vstack(activ)).astype('float16')
    
    ep_activ2d = ep_split_res(activ2d, lens)
    ws_activ2d, ns_activ2d = ep_activ2d[:50], ep_activ2d[50:]
    
    if 'activ2d' not in activ2d_data:
        activ2d_data['activ2d'] = {}
        
    if save_chk not in activ2d_data['ws']:
        activ2d_data['ws'][save_chk] = {}
    if save_chk not in activ2d_data['ns']:
        activ2d_data['ns'][save_chk] = {}
        
    activ2d_data['activ2d'][save_chk] = activ2d
    activ2d_data['ws'][save_chk]['activ2d'] = ws_activ2d
    activ2d_data['ns'][save_chk]['activ2d'] = ns_activ2d
    
    return True
    
    
    
prefixes = ['', '0.2_']
aux_tasks = ['catfacewall', 'catquad', 'catshort', 'catwall01', 'wall01']
trials = range(10)
folder = Path('data/shortcut/sc_aux_copied')

now = time.time()

for p, aux, t in itertools.product(prefixes, aux_tasks, trials):
    file = f'{p}{aux}_{t}'
    print(file)
    print('Elapsed seconds', time.time() - now)
    if not file in [i.name for i in folder.iterdir()]:
        continue
    
    activ2d_data = {'activ2d': {}, 'ws': {}, 'ns': {}}
    all_res = pickle.load(open(folder/file, 'rb'))
    for i in range(len(chk_sched)):
        collect_activ2d_from_wsns(all_res, activ2d_data, i)
    pickle.dump(activ2d_data, open(folder/(file+'_activ2d'), 'wb'))    

    