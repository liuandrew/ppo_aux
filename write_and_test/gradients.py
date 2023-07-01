import sys
sys.path.append('..')
from ppo import algo, utils
from ppo.algo.ppo import PPO, PPOAux
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from evaluation import evaluate
from tqdm import tqdm

import matplotlib.pyplot as plt
    
from read_experiments import *
from model_evaluation import *

from ppo.storage import RolloutStorage, RolloutStorageAux
from ppo.envs import make_vec_envs
from  ppo.model import Policy
import torch

import proplot as pplt
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
'''
First 4 parameters are for shared recurrent layer. Can freeze these by setting
requires_grad = False
'''

"""
Main functions - initialize_ppo_training, collect_batches_and_grads, aux_cos_sims_from_all_grads

Example: Collect batches and compare gradients generaated by RL task vs aux task
exp_name = 'nav_pdistal_batchaux/nav_pdistal_batch32auxwall0'
model, obs_rms, env_kwargs = load_model_and_env(exp_name, 0)
agent, envs, storage = initialize_ppo_training(model, obs_rms, env_kwargs=env_kwargs,
                                               num_steps=128, agent_base='DecomposeGradPPO', num_processes=16, 
                                               ppo_epoch=1, take_optimizer_step=False)
res = collect_batches_and_grads(agent, envs, storage, num_batches=10, decompose_grads=True)
cos_sims = aux_cos_sim_grad(res['all_grads'])
"""

def populate_rollouts(model, envs, rollouts, num_steps=None, seed=None, deterministic=False,
                      data_callback=None):
    """Subfunction used by collect_batches_and_grads
    
    Pass a model with envs and rollouts object to simulate environment and collect
    trajectories
    model can be loaded or initialized. envs and rollouts usually initialized from
    initialize_ppo_training

    Args:
        model (FlexBase): Neural network generating action and value outputs
        envs (VecPyTorch): Vectorized collection of environments
        rollouts (RolloutStorage): RolloutStorage object
        num_steps (_type_): _description_
        seed (_type_, optional): _description_. Defaults to None.
        deterministic (bool, optional): _description_. Defaults to False.
    """
    if seed != None:
        torch.manual_seed(seed)
        
    if num_steps == None:
        num_steps = rollouts.num_steps

    data = {}
    rollout_info = []

    for step in range(num_steps):
        #Generate rollouts for num_steps batch
        with torch.no_grad():
            outputs = model.act(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                                rollouts.masks[step])
            action = outputs['action']
            value = outputs['value']
            action_log_prob = outputs['action_log_probs']
            recurrent_hidden_states = outputs['rnn_hxs']
            auxiliary_preds = outputs['auxiliary_preds']

        obs, reward, done, infos = envs.step(action)
        # if (reward != 0).any():
        #     print(reward, done, infos)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        
        if data_callback is not None:
            data = data_callback(model, envs, recurrent_hidden_states, obs, action, 
                                 reward, done, data)
        
        auxiliary_truths = []
        for info in infos:
            if 'auxiliary' in info:
                if len(info['auxiliary']) > 0:
                    auxiliary_truths.append(info['auxiliary'])
        rollout_info.append(infos)
        if len(auxiliary_truths) > 0:
            auxiliary_truths = torch.tensor(np.vstack(auxiliary_truths))
        else:
            auxiliary_truths = None

        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks,
                        auxiliary_preds, auxiliary_truths)
        
    return data, rollout_info



def populate_rollouts_aux(model, envs, rollouts, num_steps=None, seed=None, deterministic=False,
                      data_callback=None):
    if seed != None:
        torch.manual_seed(seed)
        
    if num_steps == None:
        num_steps = rollouts.num_steps

    data = {}
    rollout_info = []

    
    for step in range(num_steps):
        #Generate rollouts for num_steps batch
        with torch.no_grad():
            outputs = model.act(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                                rollouts.masks[step])
            action = outputs['action']
            value = outputs['value']
            action_log_prob = outputs['action_log_probs']
            recurrent_hidden_states = outputs['rnn_hxs']
            auxiliary_preds = outputs['auxiliary_preds']

        obs, reward, done, infos = envs.step(action)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        
        if data_callback is not None:
            data = data_callback(model, envs, recurrent_hidden_states, obs, action, 
                                 reward, done, data)
        
        auxiliary_truths = [[] for i in range(len(model.auxiliary_output_sizes))]
        for info in infos:
            if 'auxiliary' in info and len(info['auxiliary']) > 0:
                for i, aux in enumerate(info['auxiliary']):
                    auxiliary_truths[i].append(aux)
        if len(auxiliary_truths) > 0:
            auxiliary_truths = [torch.tensor(np.vstack(aux)) for aux in auxiliary_truths]
        
        rollout_info.append(infos)
        
        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks,
                       auxiliary_preds, auxiliary_truths)
        
    return data, rollout_info
        
        
        
def clone_rollouts(copy, paste, num_steps, start=0, copy_first=True):
    """Clone all stored values from one rollout storage to another. We use this
    when seeing whether a sub-batch of a larger batch produces similar gradients, but
    mostly discarded this analysis

    Args:
        copy (RolloutStorage): Storage to copy from
        paste (RolloutStorage): Storage to paste to
        num_steps (int): Number of steps to cop
        start (int, optional): What index to start copying from. Defaults to 0.
        copy_first (bool, optional): Whether to copy the first step, since the first step
        is unique in storage (due to obs, rnn_hxs, masks being indexed +1) Usually set this to
        True if start==0. Defaults to True.
    """
    if copy_first:
        #Copy the first steps of storage types that need it
        paste.obs[0].copy_(copy.obs[start])
        paste.recurrent_hidden_states[0].copy_(copy.recurrent_hidden_states[start])
        paste.masks[0].copy_(copy.masks[start])
        paste.bad_masks[0].copy_(copy.bad_masks[start])
        
    for step in range(num_steps):
        obs = copy.obs[step+start+1]
        recurrent_hidden_states = copy.recurrent_hidden_states[step+start+1]
        action = copy.actions[step+start]
        action_log_prob = copy.action_log_probs[step+start]
        value = copy.value_preds[step+start]
        reward = copy.rewards[step+start]
        masks = copy.masks[step+start+1]
        bad_masks = copy.bad_masks[step+start+1]
        paste.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        
def update_model(agent, rollouts, use_gae=False, gamma=0.99, gae_lambda=0.95,
                 after_update=True):
    """Simple function to compute gradients from agent and stored rollouts

    Args:
        agent (PPO type): Trainer object
        rollouts (RolloutStorage): filled storage
        use_gae (bool, optional): GAE. Defaults to False.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        gae_lambda (float, optional): GAE factor. Defaults to 0.95.
        after_update (bool, optional): Whether to perform rollouts after_update step
        of resetting step count and copying obs etc. to first index. Defaults to True.

    Returns:
        _type_: _description_
    """
    #Compute last value to be used for the update
    with torch.no_grad():
        next_value = agent.actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                                            rollouts.masks[-1]).detach()
    
    rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda)
    value_loss, action_loss, dist_entropy, approx_kl, clipfracs, auxiliary_loss = agent.update(rollouts)
    
    if after_update:
        rollouts.after_update()
        
    return value_loss, action_loss, dist_entropy, approx_kl, clipfracs, auxiliary_loss



def initialize_ppo_training(model=None, obs_rms=None, env_name='NavEnv-v0', env_kwargs={}, make_env=True,
                            agent_base='LoudPPO', nn_base_kwargs={}, recurrent=True,
                            num_steps=10, num_processes=1, seed=0, ppo_epoch=4, clip_param=0.5,
                            num_mini_batch=1, value_loss_coef=0.5, entropy_coef=0.01, 
                            auxiliary_loss_coef=0.3, gamma=0.99, lr=7e-4, eps=1e-5, max_grad_norm=0.5,
                            log_dir='/tmp/gym/', device=torch.device('cpu'), 
                            capture_video=False, take_optimizer_step=True,
                            normalize=True, obs=None, aux_wrapper_kwargs={}, new_aux=True,
                            auxiliary_truth_sizes=[]):
    """Generate training objects, specifically setting up everything to generate gradients
        Important parameters:
            model, obs_rms, env_kwargs, num_steps (batch_size), num_processes, seed, 
            ppo_epoch (usually set=1), take_optimizer_step (usually set=False)

    Args:
        model (Policy, optional): Policy object (e.g. from load_model_and_env). If not provided
            generate a fresh model with nn_base and nn_base_kwargs
        obs_rms (RunningMeanStd, optional): obs_rms object for vectorized envs. Defaults to None.
        env_name (str, optional): Defaults to 'NavEnv-v0'.
        env_kwargs (dict, optional): Defaults to {}.
        nn_base (str, optional): Used to create model if model is not provided. 
            Defaults to 'FlexBase'.
        agent_base (str, optional): Used to create trainer object. Defaults to 'LoudPPO',
            can also use 'PPO' and 'DecomposeGradPPO'.
        nn_base_kwargs (dict, optional): Used to create model if model is not provided. 
            Defaults to {}.
        recurrent (bool, optional): Used if model==None. Defaults to True.
        num_steps (int, optional): Batch size to use. Defaults to 10.
        num_processes (int, optional): Number of concurrent processes. Defaults to 1.
        seed (int, optional): Randomizer seed. Defaults to 0.
        ppo_epoch (int, optional): Number of epochs to run for PPO. Defaults to 4. Usually
            we will want to set this to 1 to collect grads with
        clip_param (float, optional): PPO clip param. Defaults to 0.5.
        num_mini_batch (int, optional): Number of minibatches to split training rollouts into. 
            Defaults to 1.
        value_loss_coef (float, optional): Value loss weighting. Defaults to 0.5.
        entropy_coef (float, optional): Entropy loss weighting. Defaults to 0.01.
        auxiliary_loss_coef (float, optional): Auxiliary loss weighting. Defaults to 0.3.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        lr (_type_, optional): Learning rate. Defaults to 7e-4.
        eps (_type_, optional): _description_. Defaults to 1e-5.
        max_grad_norm (float, optional): Cap on gradient steps. Defaults to 0.5.
        log_dir (str, optional): Logging directory. Defaults to '/tmp/gym/'.
        device (_type_, optional): Device to run on. Defaults to torch.device('cpu').
        capture_video (bool, optional): Whether to capture video on episodes. Defaults to False.
        take_optimizer_step (bool, optional): Whether to actually take gradient update
            step. Defaults to True.
        normalize (bool, optional): Whether to normalize vectorized environment observations. 
            Defaults to True.
        obs (torch.Tensor, optional): Need to pass the first observation if not making new environments

    Returns:
        agent, envs, rollouts
    """
    
    #Initialize vectorized environments
    # envs = make_vec_envs(env_name, seed, num_processes, gamma, log_dir, device, False,
    #                      capture_video=capture_video, env_kwargs=env_kwargs)
    if make_env:
        envs = make_vec_envs(env_name, seed, num_processes, gamma, log_dir, device, False,
                            capture_video=capture_video, env_kwargs=env_kwargs, normalize=normalize,
                            **aux_wrapper_kwargs)
    else:
        envs = None

    env = gym.make('NavEnv-v0', **env_kwargs)

    if model is None:
        if new_aux:
            nn_base = 'FlexBaseAux'
        else:
            nn_base = 'FlexBase'
        model = Policy(env.observation_space.shape,
                       env.action_space,
                       base=nn_base,
                       base_kwargs={'recurrent': recurrent,
                           **nn_base_kwargs})
        model.to(device)
    
    #Wrap model with an agent algorithm object
    # agent = algo.PPO(model, clip_param, ppo_epoch, num_mini_batch,
    try:
        # if new_aux:
        #     agent = PPOAux(model, clip_param, ppo_epoch, num_mini_batch,
        #             value_loss_coef, entropy_coef, auxiliary_loss_coef, lr=lr,
        #             eps=eps, max_grad_norm=max_grad_norm)
        # else:
        base = globals()[agent_base]
        agent = base(model, clip_param, ppo_epoch, num_mini_batch,
                        value_loss_coef, entropy_coef, auxiliary_loss_coef, lr=lr,
                        eps=eps, max_grad_norm=max_grad_norm,
                        take_optimizer_step=take_optimizer_step)
    except:
        print('Model type not found')
        return False


    #Initialize storage
    if new_aux:
        rollouts = RolloutStorageAux(num_steps, num_processes, env.observation_space.shape, env.action_space,
                            model.recurrent_hidden_state_size, model.auxiliary_output_sizes,
                            auxiliary_truth_sizes)
    else:
        rollouts = RolloutStorage(num_steps, num_processes, env.observation_space.shape, env.action_space,
                                model.recurrent_hidden_state_size, model.auxiliary_output_size)
    #Storage objects initializes a bunch of empty tensors to store information, e.g.
    #obs has shape (num_steps+1, num_processes, obs_shape)
    #rewards has shape (num_steps, num_processes, 1)
    

    #If loading a previously trained model, pass an obs_rms object to set the vec envs to use
    
    if normalize and obs_rms != None:
        vec_norm = utils.get_vec_normalize(envs)
        if vec_norm is not None and obs_rms is not None:
            vec_norm.obs_rms = obs_rms

        
    #obs, recurrent_hidden_states, value_preds, returns all have batch size num_steps+1
    #rewards, action_log_probs, actions, masks, auxiliary_preds, auxiliary_truths all have batch size num_steps
    if make_env:
        obs = envs.reset()
    elif obs == None:
        raise Exception('No obs passed and no env created, storage cannot be initialized')
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    
    return agent, envs, rollouts


def generate_storage(model, num_steps, num_processes, obs, env=None, env_name='NavEnv-v0', env_kwargs={}):
    """Mini function to create a RolloutStorage without generating new vectorized envs

    Args:
        model (Policy): model object to get rnn_hxs and aux_output sizes from.
        num_steps (int): batch size expected
        num_processes (int): number of processes expected
        obs (torch.Tensor): first observation to populate storage wit
        env (gym.Env, optional): Single gym env to get obs and action space from. If not given
        will attempt to create an env to find these using env_name and env_kwargs. Defaults to None.
        env_name (str, optional): Gym env name. Defaults to 'NavEnv-v0'.
        env_kwargs (dict, optional): Gym env kwargs. Defaults to {}.

    Returns:
        RolloutStorage: storage object
    """
    if env == None:
        env = gym.make(env_name, **env_kwargs)
    
    rollouts = RolloutStorage(num_steps, num_processes, env.observation_space.shape, env.action_space,
                          model.recurrent_hidden_state_size, model.auxiliary_output_size)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    
    return rollouts



def collect_batches_and_grads(agent, envs, rollouts, num_batches=1, seed=None, num_layers=None,
                             decompose_grads=False, data_callback=None, new_aux=False):
    """Main function to collect batches and grad

    Args:
        agent (PPO type): Trainer object 
        envs (VecPyTorch): Collection of vectorized envs
        rollouts (RolloutStorage): Storage object 
        num_batches (int, optional): How many batches to collect. Defaults to 1.
        seed (int, optional): Randomizing seed. Defaults to None.
        num_layers (int, optional): How many sets of parameters to collect. If not
        given, collect all. Defaults to None.
        decompose_grads (bool, optional): Whether to separate out loss grads (actor,
        value, entropy, auxiliary). agent must be a DecomposeGradPPO object to use this. 
        Defaults to False.

    Returns:
        dict: results
            all_grads: gradients indexed by all_grads[param_set][batch] (Tensor)
                If decompose_grads==True, this a dict indexed by
                all_grads[grad_type][param_set][batch]
            rewarded: whether each batch had reward or not rewarded[batch]
            value_losses: losses for values value_losses[batch]
            action_losses: losses for actions action_losses[batch]
            rewards: total rewards in each batch rewrds[batch]
            value_diff: total sum difference between true and expected values value_diff[batch]
    """
    rewarded = []
    value_losses = []
    action_losses = []
    auxiliary_losses = []
    num_rewards = []
    value_diffs = []
    datas = []
    infos = []

    params = list(agent.actor_critic.parameters())
    if num_layers == None:
        num_layers = len(list(params))

    if decompose_grads:
        grad_types = ['value', 'action', 'auxiliary', 'entropy']
        all_grads = {name: [] for name in grad_types}
        if agent.ppo_epoch > 1:
            for name in grad_types:
                for e in range(agent.ppo_epoch):
                    all_grads[name].append([])
                    for i in range(num_layers):
                        all_grads[name][e].append([])
        else:
            for name in grad_types:
                for i in range(num_layers):
                    all_grads[name].append([])
    else:
        all_grads = []
        if agent.ppo_epoch > 1:
            for e in range(agent.ppo_epoch):
                all_grads.append([])
                for i in range(num_layers):
                    all_grads[e].append([])
        else:
            for i in range(num_layers):
                all_grads.append([])
            
    
    for n in range(num_batches):
        if new_aux:
            data, info = populate_rollouts_aux(agent.actor_critic, envs, rollouts, rollouts.num_steps, seed=seed,
                                    data_callback=data_callback)            
        else:            
            data, info = populate_rollouts(agent.actor_critic, envs, rollouts, rollouts.num_steps, seed=seed,
                                    data_callback=data_callback)
        
        agent.optimizer.zero_grad()
        next_value = agent.actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                                     rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, False, 0.99, 0.95)
        
        
        if decompose_grads:
            value_loss, action_loss, dist_entropy, approx_kl, clipfracs, auxiliary_loss, \
                grads = agent.update(rollouts)
            if agent.ppo_epoch > 1:
                for name in grad_types:
                    for e in range(agent.ppo_epoch):
                        for i in range(num_layers):
                            all_grads[name][e][i].append(grads[e][name][i])
            else:
                for name in grad_types:
                    for i in range(num_layers):
                        all_grads[name][i].append(grads[name][i])
        else:
            value_loss, action_loss, dist_entropy, approx_kl, clipfracs, auxiliary_loss, \
                all_grads = agent.update(rollouts)
            
        rew = rollouts.rewards
        rewarded.append((rew != 0).any().item())
        num_rewards.append(len(rew[rew != 0]))
        value_losses.append(value_loss)
        action_losses.append(action_loss)
        auxiliary_losses.append(auxiliary_loss)
        datas.append(data)
        infos.append(info)
        
        value_diff = torch.sum(rollouts.returns - rollouts.value_preds)
        value_diffs.append(value_diff.item())
        rollouts.after_update()
    
    return {
        'all_grads': all_grads,
        'rewarded': rewarded,
        'value_losses': value_losses,
        'action_losses': action_losses,
        'auxiliary_losses': auxiliary_losses,
        'rewards': num_rewards,
        'value_diff': value_diffs,
        'data': datas
    }



def collect_batches_and_auxrew_grads(agent, envs, rollouts, num_batches=1, seed=None, num_layers=None,
                             decompose_grads=False, data_callback=None, compute_pure_bonus=False):
    """Main function to collect batches and grad for a bonus reward auxiliary task. 
        Assume envs includes some bonus reward

    Args:
        agent (PPO type): Trainer object 
        envs (VecPyTorch): Collection of vectorized envs
        rollouts (RolloutStorage): Storage object 
        num_batches (int, optional): How many batches to collect. Defaults to 1.
        seed (int, optional): Randomizing seed. Defaults to None.
        num_layers (int, optional): How many sets of parameters to collect. If not
        given, collect all. Defaults to None.
        decompose_grads (bool, optional): Whether to separate out loss grads (actor,
        value, entropy, auxiliary). agent must be a DecomposeGradPPO object to use this. 
        Defaults to False.

    Returns:
        dict: results
            all_grads: gradients indexed by all_grads[param_set][batch] (Tensor)
                If decompose_grads==True, this a dict indexed by
                all_grads[grad_type][param_set][batch]
            rewarded: whether each batch had reward or not rewarded[batch]
            value_losses: losses for values value_losses[batch]
            action_losses: losses for actions action_losses[batch]
            rewards: total rewards in each batch rewrds[batch]
            value_diff: total sum difference between true and expected values value_diff[batch]
    """
    rewarded = []
    value_losses = []
    action_losses = []
    num_rewards = []
    value_diffs = []
    all_rewards_bonus = []
    all_rewards_goal = []
    datas = []
    infos = []
    
    params = list(agent.actor_critic.parameters())
    if num_layers == None:
        num_layers = len(list(params))

    if decompose_grads:
        grad_types = ['value', 'action', 'auxiliary', 'entropy']
        all_grads_bonus = {name: [] for name in grad_types}
        all_grads_goal = {name: [] for name in grad_types}
        all_grads_pure_bonus = {name: [] for name in grad_types}
        for name in grad_types:
            for i in range(num_layers):
                all_grads_bonus[name].append([])
                all_grads_goal[name].append([])
                all_grads_pure_bonus[name].append([])
    else:
        all_grads_bonus = []
        all_grads_goal = []
        all_grads_pure_bonus = []
        for i in range(num_layers):
            all_grads_bonus.append([])
            all_grads_goal.append([])
            all_grads_pure_bonus.append([])
        
    
    for n in range(num_batches):
        data, info = populate_rollouts(agent.actor_critic, envs, rollouts, rollouts.num_steps, seed=seed,
                                 data_callback=data_callback)
        
        #First set of gradient calculations with the original reward
        agent.optimizer.zero_grad()
        next_value = agent.actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                                     rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, False, 0.99, 0.95)
        
        if decompose_grads:
            value_loss, action_loss, dist_entropy, approx_kl, clipfracs, auxiliary_loss, \
                grads = agent.update(rollouts)
            
            for name in grad_types:
                for i in range(num_layers):
                    all_grads_bonus[name][i].append(grads[name][i])
        else:
            value_loss, action_loss, dist_entropy, approx_kl, clipfracs, auxiliary_loss, \
                grads = agent.update(rollouts)

            for i in range(num_layers):
                all_grads_bonus[i].append(params[i].grad.clone())
                
        #Second set of gradient calculations removing any bonus rewards smaller than 1
        rew_goal = rollouts.rewards.clone()
        rew_pure_bonus = rollouts.rewards.clone()
        
        all_rewards_bonus.append(rew_goal.clone())
        rew_goal[rew_goal < 1.0] -= rew_goal[rew_goal < 1.0]
        rollouts.rewards.copy_(rew_goal)
        
        agent.optimizer.zero_grad()
        next_value = agent.actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                                     rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, False, 0.99, 0.95)
        if decompose_grads:
            value_loss, action_loss, dist_entropy, approx_kl, clipfracs, auxiliary_loss, \
                grads = agent.update(rollouts)
            
            for name in grad_types:
                for i in range(num_layers):
                    all_grads_bonus[name][i].append(grads[name][i])
        else:
            value_loss, action_loss, dist_entropy, approx_kl, clipfracs, auxiliary_loss, \
                grads = agent.update(rollouts)

            for i in range(num_layers):
                all_grads_goal[i].append(params[i].grad.clone())
                
        #Third set of gradient calculations removing the goal alone
        if compute_pure_bonus:
            rew_pure_bonus[rew_pure_bonus >= 1.0] -= rew_pure_bonus[rew_pure_bonus >= 1.0]
            rollouts.rewards.copy_(rew_pure_bonus)
            
            agent.optimizer.zero_grad()
            next_value = agent.actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                                        rollouts.masks[-1]).detach()
            rollouts.compute_returns(next_value, False, 0.99, 0.95)
            if decompose_grads:
                value_loss, action_loss, dist_entropy, approx_kl, clipfracs, auxiliary_loss, \
                    grads = agent.update(rollouts)
                
                for name in grad_types:
                    for i in range(num_layers):
                        all_grads_bonus[name][i].append(grads[name][i])
            else:
                value_loss, action_loss, dist_entropy, approx_kl, clipfracs, auxiliary_loss, \
                    grads = agent.update(rollouts)

                for i in range(num_layers):
                    all_grads_pure_bonus[i].append(params[i].grad.clone())
            
        rew = rollouts.rewards.clone()
        all_rewards_goal.append(rew)
        rewarded.append((rew != 0).any().item())
        num_rewards.append(len(rew[rew != 0]))
        value_losses.append(value_loss)
        action_losses.append(action_loss)
        datas.append(data)
        infos.append(info)
        
        value_diff = torch.sum(rollouts.returns - rollouts.value_preds)
        value_diffs.append(value_diff.item())
        rollouts.after_update()
    
    return {
        'all_grads_bonus': all_grads_bonus,
        'all_grads_goal': all_grads_goal,
        'all_grads_pure_bonus': all_grads_pure_bonus,
        'all_rewards_goal': all_rewards_goal,
        'all_rewards_bonus': all_rewards_bonus,
        'rewarded': rewarded,
        'value_losses': value_losses,
        'action_losses': action_losses,
        'rewards': num_rewards,
        'value_diff': value_diffs,
        'data': datas,
        'info': infos
    }



def aux_cos_sims_from_all_grads(all_grads, use_layer_subset=True, actor_layer_subset=False,
                                pairwise=False):
    """This function assumes 
    
    From an all_grads object which has auxiliary and RL grads,
    compute the cosine similarities between the gradients 
    

    Args:
        all_grads (dict): Dictionary of gradients to be used
        use_layer_subset (bool, optional): If set to True, only compute gradients
            for specific layers affected by auxiliary gradient. Will assume these are the
            actor layer by default, but switch to critic layer if the auxiliary task
            key ends in a c (e.g., all_grad['goaldistc']). Defaults to True.
        actor_layer_subset (bool, optional - NOT USED): Used for if we are not indexing the all_grads
            dict by auxiliary task (e.g., indexing by trial). If this is True, force use
            of actor test layers for comparisons (if use_layer_subset is True)
        pairwise (bool, optional): If set to True, compute the pairwise cosine similarities
            between gradients for each batch to each other. Otherwise, only compute cosine
            similarity for aux and RL gradients coming from the same batch. Defaults to False.

    Returns:
        dict: Results with 'x', 'y' (cos_sim means), 'std', 'low' (12.5 percentile), 
            'high' (87.5 percentile), 'c' (colors)
            Each dict entry will be a list of lists. The list indexes by [aux][batch]
            e.g., y[0][3] will give the mean cos_sims for aux_tasks[0] ('goaldist') and
                3rd batch
    """

    num_layers = 16
    
    # Colors to assign to auxiliary tasks (they will be assigned in order)
    colors = pplt.Cycle('default').by_key()['color']
    hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    rgb_colors = np.array([hex_to_rgb(color) for color in colors])/255

    
    # Assume the first iterator of all_grads is a dict
    all_xs = []
    all_ys = []
    all_lows = []
    all_highs = []
    all_stds = []
    all_cs = []
    
    for n, key in enumerate(all_grads):
        xs = []
        ys = []
        lows = []
        stds = []
        highs = []
        colors = []
        
        # Next layer of all_grads may be dict or list
        if type(all_grads[key]) == dict:
            grads = [all_grads[key][key2] for key2 in all_grads[key]]
        elif type(all_grads[key]) == list:
            grads = all_grads[key]
        
        if use_layer_subset:
            if type(key) == str and key[-1] == 'c':
                # Auxiliary head placed on critic side
                layers = [0, 1, 4, 8]
            else:
                # Auxiliary head place on actor side
                layers = [0, 1, 6, 10]
        else:
            layers = range(num_layers)
                
        for x, grad in enumerate(grads):
            cos_sims = []
            
            for i in layers:
                # Combine gradients and calculate cosine similarities
                value_grad = torch.vstack([g.reshape(-1) for g in grad['value'][i]])
                action_grad = torch.vstack([g.reshape(-1) for g in grad['action'][i]])
                entropy_grad = torch.vstack([g.reshape(-1) for g in grad['entropy'][i]])
                aux_grad = torch.vstack([g.reshape(-1) for g in grad['auxiliary'][i]])
                cumu_grad = value_grad + action_grad + entropy_grad
                
                if pairwise:
                    # If pairwise is True, compare every batch to every other batch gradient
                    cs = np.triu(cosine_similarity(cumu_grad, aux_grad), k=1)
                    cs = cs[cs != 0]
                    cos_sims += list(cs)
                else:
                    # Otherwise only compare each gradient to its counterpart in the same batch
                    for j in range(len(cumu_grad)):
                        cs = cosine_similarity(cumu_grad[j].reshape(1, -1),
                                               aux_grad[j].reshape(1, -1))
                        cos_sims += list(cs)
            
            xs.append(x)
            ys.append(np.mean(cos_sims))
            stds.append(np.std(cos_sims))
            lows.append(np.percentile(cos_sims, 12.5))
            highs.append(np.percentile(cos_sims, 87.5))
            colors.append(rgb_colors[n])
            
        all_xs.append(xs)
        all_ys.append(ys)
        all_stds.append(stds)
        all_lows.append(lows)
        all_highs.append(highs)
        all_cs.append(colors)
        
    return {
        'x': all_xs,
        'y': all_ys,
        'std': all_stds,
        'low': all_lows,
        'high': all_highs,
        'c': all_cs
    }
            
            
            
def aux_cos_sim_grad(grads, use_layer_subset='a', pairwise=False):
    """Compare pred aux grads vs rl grads for a single agent between batches

    Args:
        grads (all_grads): all_grads indexed by all_grads[type][layer][batch]
        use_layer_subset (str, optional): Whether to use specific subset of layers
        'a': actor layers, 'c': critic layers, None/False: all layers
        pairwise (bool, optional): Whether to compare each batch to each other. 
        Defaults to False.

    Returns:
        cos_sims (list): cosine similarities
    """
    if use_layer_subset == 'a':
        layers = [0, 1, 6, 10]
    elif use_layer_subset == 'c':
        layers = [0, 1, 4, 8]
    elif type(use_layer_subset) == list:
        layers = use_layer_subset
    elif use_layer_subset == None or use_layer_subset == False:
        layers = range(len(grads['action']))

    cos_sims = []
    for i in layers:
        # Combine gradients and calculate cosine similarities
        value_grad = torch.vstack([g.reshape(-1) for g in grads['value'][i]])
        action_grad = torch.vstack([g.reshape(-1) for g in grads['action'][i]])
        entropy_grad = torch.vstack([g.reshape(-1) for g in grads['entropy'][i]])
        aux_grad = torch.vstack([g.reshape(-1) for g in grads['auxiliary'][i]])
        cumu_grad = value_grad + action_grad + entropy_grad

        if pairwise:
            # If pairwise is True, compare every batch to every other batch gradient
            cs = np.triu(cosine_similarity(cumu_grad, aux_grad), k=1)
            cs = cs[cs != 0]
            cos_sims += list(cs)
            # print(cs)
        else:
            # Otherwise only compare each gradient to its counterpart in the same batch
            for j in range(len(cumu_grad)):
                cs = cosine_similarity(cumu_grad[j].reshape(1, -1),
                                       aux_grad[j].reshape(1, -1)).reshape(-1)
                cos_sims += list(cs)
                
    return cos_sims




def cos_sim_grad(grads1, grads2, use_layer_subset=None, pairwise=True,
                 decompose_grads=False):
    """Compare grads between 2 sets of grads. The same grad can be used for
    both entries to do a within agent comparison

    Args:
        grads1 (all_grads): all_grads indexed by all_grads[layer][batch]
        grads2 (all_grads): all_grads indexed by all_grads[layer][batch]
        use_layer_subset (str, optional): Whether to use specific subset of layers
        'a': actor layers, 'c': critic layers, None/False: all layers
        pairwise (bool, optional): Whether to compare each batch to each other. 
        Defaults to True.
        decompose_grads (str, optional): Whether we expect to have each grad set
            decomposed. If None/False, we expect all_grads to directly have cumulative
            gradient (all_grads[layer][batch]), otherwise expect decomposed
            all_grads[type][layer][batch]
            Options:
            'typewise': Compare each type to its counterpart
            'rlvsaux': Compare RL grads (actor+value+entropy) and aux grads
            'collect': Compare complete grad (actorvalue+entropy+aux)
            None/False: Assume we are passed cumulated grad already

    Returns:
        cos_sims (list): cosine similarities
    """
    if use_layer_subset == None or use_layer_subset == False:
        if decompose_grads:
            layers1 = len(grads1['action'])
            layers2 = len(grads2['action'])
            layers = range(min(layers1, layers2))
        else:
            layers1 = len(grads1)
            layers2 = len(grads2)
            layers = range(min(layers1, layers2))
    else:
        if type(use_layer_subset) == str:
            if use_layer_subset == 'a':
                layers = [0, 1, 6, 10]
            elif use_layer_subset == 'c':
                layers = [0, 1, 4, 8]
        elif type(use_layer_subset) == list:            
            layers = use_layer_subset
            
    cos_sims = []
    # print(layers)
    for i in layers:
        compare_grads = []
        
        if decompose_grads != None and decompose_grads != False:
            value_grad1 = torch.vstack([g.reshape(-1) for g in grads1['value'][i]])
            action_grad1 = torch.vstack([g.reshape(-1) for g in grads1['action'][i]])
            entropy_grad1 = torch.vstack([g.reshape(-1) for g in grads1['entropy'][i]])
            aux_grad1 = torch.vstack([g.reshape(-1) for g in grads1['auxiliary'][i]])

            value_grad2 = torch.vstack([g.reshape(-1) for g in grads2['value'][i]])
            action_grad2 = torch.vstack([g.reshape(-1) for g in grads2['action'][i]])
            entropy_grad2 = torch.vstack([g.reshape(-1) for g in grads2['entropy'][i]])
            aux_grad2 = torch.vstack([g.reshape(-1) for g in grads2['auxiliary'][i]])
        if decompose_grads == 'typewise':
            compare_grads = [
                [value_grad1, value_grad2],
                [action_grad1, action_grad2],
                [entropy_grad1, entropy_grad2],
                [aux_grad1, aux_grad2],
            ]
        elif decompose_grads == 'rlvsaux':
            cumu_grad1 = value_grad1 + action_grad1 + entropy_grad1
            cumu_grad2 = value_grad2 + action_grad2 + entropy_grad2
            compare_grads = [
                [cumu_grad1, cumu_grad2],
                [aux_grad1, aux_grad2],
            ]
        elif decompose_grads == 'collect':
            cumu_grad1 = value_grad1 + action_grad1 + entropy_grad1 + aux_grad1
            cumu_grad2 = value_grad2 + action_grad2 + entropy_grad2 + aux_grad2            
            compare_grads = [
                [cumu_grad1, cumu_grad2],
            ]
        else:
            cumu_grad1 = torch.vstack([g.reshape(-1) for g in grads1[i]])
            cumu_grad2 = torch.vstack([g.reshape(-1) for g in grads2[i]])
            compare_grads = [
                [cumu_grad1, cumu_grad2],
            ]
        
        for grad_pair in compare_grads:
            grad1 = grad_pair[0]
            grad2 = grad_pair[1]
            if pairwise:
                # If pairwise is True, compare every batch to every other batch gradient
                # cs = np.triu(cosine_similarity(grad1, grad2), k=1)
                # cs = cs[cs != 0]
                cs = cosine_similarity(grad1, grad2)
                cos_sims.append(cs)
            else:
                # Otherwise only compare each gradient to its counterpart in the same batch
                for j in range(len(grad1)):
                    cs = cosine_similarity(grad1[j].reshape(1, -1),
                                           grad2[j].reshape(1, -1)).reshape(-1)
                    cos_sims += list(cs)
                    
    return cos_sims



def cos_sim_grad_layerwise(grads1, grads2):
    """
    Compare grads of two sets, both indexed by grads[layer][batch]
    
    Return cosine similarities indexed by layer
    """
    

          

"""
Trainer Class Variations
These objects are usually used to produce gradients from a rollouts object
on a model and then perform gradient update steps. The variations here allow
us to backprop and generate gradients without actually taking the update step
LoudPPO does exactly that, while DecomposeGradPPO also optionally separates
policy, value, entropy, and auxiliary gradients
"""  
            
            
class LoudPPO():
    '''
    Variation on our trainer object that allows for printing of 
    some debugging information and more importantly allows us to compute
    gradients without taking the actual update step to network weights
    '''
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 auxiliary_loss_coef=0,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 verbose=False,
                 take_optimizer_step=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.auxiliary_loss_coef = auxiliary_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        
        #Additional parameter that allows us to skip taking optimizer step so that
        #we can keep trajectories same for different batch sizes
        self.take_optimizer_step = take_optimizer_step
        self.verbose = verbose

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        auxiliary_loss_epoch = 0

        clipfracs = []
        explained_vars = []
        
        num_update_steps = 0
        grads = []
        for e in range(self.ppo_epoch):
            grads.append([])
            
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                num_update_steps += 1
                if self.verbose:
                    print(num_update_steps)
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ, auxiliary_pred_batch, auxiliary_truth_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, auxiliary_preds = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                logratio = action_log_probs - old_action_log_probs_batch 
                ratio = torch.exp(logratio)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                #Andy: compute approx kl
                with torch.no_grad():
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                    ]

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                
                if self.actor_critic.has_auxiliary:
                    auxiliary_loss = 0.5 * (auxiliary_truth_batch - auxiliary_preds).pow(2).mean()
                else:
                    auxiliary_loss = torch.zeros(1)
                # print(auxiliary_truth_batch, auxiliary_preds, auxiliary_loss)


                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss + 
                 auxiliary_loss * self.auxiliary_loss_coef -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                if self.take_optimizer_step:
                    self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                auxiliary_loss_epoch += auxiliary_loss.item()
                
                params = list(self.actor_critic.parameters())
                num_param_layers = len(params)
                for param in params:
                    if param.grad == None:
                        grad = torch.zeros(param.shape)
                    else:
                        grad = param.grad.clone()
                    grads[e].append(grad)
            
            
                

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates



        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, \
            approx_kl, clipfracs, auxiliary_loss_epoch, grads

    
    
class DecomposeGradPPO():
    '''
    Variation on our trainer object that further keeps track of grads induced
    by each individual loss and returns them on the update function
    Note this is going to assume that we only have one single ppo_epoch and one
    minibatch 
    '''
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 auxiliary_loss_coef=0,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 verbose=False,
                 take_optimizer_step=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.auxiliary_loss_coef = auxiliary_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        
        #Additional parameter that allows us to skip taking optimizer step so that
        #we can keep trajectories same for different batch sizes
        self.take_optimizer_step = take_optimizer_step
        self.verbose = verbose

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        auxiliary_loss_epoch = 0

        clipfracs = []
        explained_vars = []
        
        num_update_steps = 0
        
        grads = []
        for e in range(self.ppo_epoch):
            grads.append(defaultdict(list))

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                num_update_steps += 1
                if self.verbose:
                    print(num_update_steps)
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ, auxiliary_pred_batch, auxiliary_truth_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, auxiliary_preds = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                logratio = action_log_probs - old_action_log_probs_batch 
                ratio = torch.exp(logratio)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                #Andy: compute approx kl
                with torch.no_grad():
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                    ]

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                
                if self.actor_critic.has_auxiliary:
                    auxiliary_loss = 0.5 * (auxiliary_truth_batch - auxiliary_preds).pow(2).mean()
                else:
                    auxiliary_loss = torch.zeros(1)
                # print(auxiliary_truth_batch, auxiliary_preds, auxiliary_loss)


                # (value_loss * self.value_loss_coef + action_loss + 
                #  auxiliary_loss * self.auxiliary_loss_coef -
                #  dist_entropy * self.entropy_coef).backward()             
                params = list(self.actor_critic.parameters())
                num_param_layers = len(params)

                loss_names = ['value', 'action', 'auxiliary', 'entropy']
                losses = [
                    value_loss*self.value_loss_coef,
                    action_loss,
                    auxiliary_loss*self.auxiliary_loss_coef,
                    -dist_entropy
                ]
                
                for i, loss in enumerate(losses):
                    name = loss_names[i]
                    self.optimizer.zero_grad()
                    if loss.grad_fn != None:
                        loss.backward(retain_graph=True)
                    for param in params:
                        if param.grad == None:
                            grad = torch.zeros(param.shape)
                        else:
                            grad = param.grad.clone()
                        grads[e][name].append(grad)
                                
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                if self.take_optimizer_step:
                    self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                auxiliary_loss_epoch += auxiliary_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        
        if len(grads) == 1:
            grads = grads[0]



        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, \
            approx_kl, clipfracs, auxiliary_loss_epoch, grads
            
            
            
            
class DecomposeGradPPOAux():
    '''
    Variation on our trainer object that further keeps track of grads induced
    by each individual loss and returns them on the update function
    Note this is going to assume that we only have one single ppo_epoch and one
    minibatch 
    '''
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 auxiliary_loss_coef=0,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 verbose=False,
                 take_optimizer_step=True):

        self.actor_critic = actor_critic
        self.auxiliary_types = actor_critic.base.auxiliary_layer_types
        self.cross_entropy_loss = nn.CrossEntropyLoss() #used for multiclass aux loss

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.auxiliary_loss_coef = auxiliary_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        
        #Additional parameter that allows us to skip taking optimizer step so that
        #we can keep trajectories same for different batch sizes
        self.take_optimizer_step = take_optimizer_step
        self.verbose = verbose

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        auxiliary_loss_epoch = 0

        clipfracs = []
        explained_vars = []
        
        num_update_steps = 0
        
        grads = []
        for e in range(self.ppo_epoch):
            grads.append(defaultdict(list))
            
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                num_update_steps += 1
                if self.verbose:
                    print(num_update_steps)
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ, auxiliary_pred_batch, auxiliary_truth_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, auxiliary_preds = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                logratio = action_log_probs - old_action_log_probs_batch 
                ratio = torch.exp(logratio)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                #Andy: compute approx kl
                with torch.no_grad():
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                    ]

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                
                if self.actor_critic.has_auxiliary:
                    auxiliary_losses = torch.zeros(len(self.auxiliary_types))
                    for i, aux_type in enumerate(self.auxiliary_types):
                        if aux_type == 0:
                            # auxiliary_loss += 0.5 * (auxiliary_truth_batch[i] - auxiliary_preds[i]).pow(2).mean()
                            auxiliary_losses[i] += 0.5 * (auxiliary_truth_batch[i] - auxiliary_preds[i]).pow(2).mean()
                        elif aux_type == 1:
                            # auxiliary_loss += self.cross_entropy_loss(auxiliary_preds[i], auxiliary_truth_batch[i])
                            auxiliary_losses[i] += self.cross_entropy_loss(auxiliary_preds[i], auxiliary_truth_batch[i].long().squeeze())
                    auxiliary_loss = auxiliary_losses.sum()
                else:
                    auxiliary_loss = torch.zeros(1)
                # print(auxiliary_truth_batch, auxiliary_preds, auxiliary_loss)


                # (value_loss * self.value_loss_coef + action_loss + 
                #  auxiliary_loss * self.auxiliary_loss_coef -
                #  dist_entropy * self.entropy_coef).backward()             
                params = list(self.actor_critic.parameters())
                num_param_layers = len(params)

                loss_names = ['value', 'action', 'auxiliary', 'entropy']
                losses = [
                    value_loss*self.value_loss_coef,
                    action_loss,
                    auxiliary_loss*self.auxiliary_loss_coef,
                    -dist_entropy*self.entropy_coef
                ]
                
                for i, loss in enumerate(losses):
                    name = loss_names[i]
                    self.optimizer.zero_grad()
                    if loss.grad_fn != None:
                        loss.backward(retain_graph=True)
                    for param in params:
                        if param.grad == None:
                            grad = torch.zeros(param.shape)
                        else:
                            grad = param.grad.clone()
                        grads[e][name].append(grad)
                                
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                if self.take_optimizer_step:
                    self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                auxiliary_loss_epoch += auxiliary_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        if len(grads) == 1:
            grads = grads[0]

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, \
            approx_kl, clipfracs, auxiliary_loss_epoch, grads