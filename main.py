import copy
import glob
import os
import time
from collections import deque
from shutil import copyfile

import gym
import gym_nav
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo.ppo import PPOAux
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage, RolloutStorageAux
from evaluation import evaluate

from scheduler import write_latest_exp_complete


def main():
    args = get_args()

    #Andy: setup W&B and tensorboard
    if args.exp_name is not None:
        run_name = f"{args.exp_name}__{int(time.time())}"
    else:
        run_name = f"{args.env_name}__{args.seed}__{int(time.time())}"

    print(run_name) #print run name for logging
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        logdir = wandb.run.dir
        writer = SummaryWriter(logdir)
    else:
        writer = SummaryWriter(f"runs/{run_name}")

    #Setup video uploading
    vid_dir = 'video'

    for f in os.listdir(vid_dir):
        os.remove(os.path.join(vid_dir, f))


    # def file_available(f):
    #     if os.path.exists(f):
    #         try:
    #             os.rename(f, f)
    #             return True
    #         except OSError as e:
    #             return False

    def available_videos():
        vid_dir = 'video'
        unuploaded_vids = []
        idxs = []
        for f in os.listdir(vid_dir):
            if '.mp4' in f and f not in logged_videos:
                unuploaded_vids.append(f)

        if len(unuploaded_vids) > 1:
            for f in unuploaded_vids:
                idxs.append(int(f.split('.mp4')[0].split('-')[-1]))
            f = unuploaded_vids[np.argmin(idxs)]
            return os.path.join(vid_dir, f), f
        else:
            return False

    def upload_videos():
        available = available_videos()
        if args.upload_video:
            if available is not False:
                path = available[0]
                f = available[1]
                wandb.log({'video': wandb.Video(path),
                    'format': 'gif'})
                logged_videos.append(f)
        else:
            save_path = 'saved_video/' + args.exp_name
            #move videos to another folder
            if len(logged_videos) == 0 and args.exp_name not in os.listdir('saved_video'):
                os.mkdir(save_path)
            if available is not False:
                path = available[0]
                f = available[1]
                copyfile(path, os.path.join(save_path, f))
                logged_videos.append(f)



    logged_videos = []
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    #Perform some checks for new auxiliary task trainers
    if args.use_new_aux == True:
        print('New Auxiliary training methods')
        print('auxiliary truth sizes: ' + str(args.auxiliary_truth_sizes))
        print('auxiliary heads: ' + str(args.nn_base_kwargs['auxiliary_heads']))
        if len(args.auxiliary_truth_sizes) != len(args.nn_base_kwargs['auxiliary_heads']):
            raise Exception('number of auxiliary_truth_sizes should be equivalent to number of auxiliary heads')
        if args.nn_base != 'FlexBaseAux':
            print('WARNING: nn_base should probably be FlexBaseAux for new aux methods')

    # Andy: generate save path ahead of time
    if args.save_subdir is not None:
        save_path = os.path.join(args.save_dir, args.save_subdir)
        try:
            os.makedirs(save_path)
        except OSError:
            pass
    else:
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

    if args.save_name is not None:
        save_path = os.path.join(save_path, args.save_name + '.pt')
    else:
        save_path = os.path.join(save_path, args.env_name)


    # Andy: generate path for saving checkpoints
    if args.checkpoint_interval > 0:
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        try:
            os.makedirs(checkpoint_path)
        except OSError:
            pass

        # Generate sub-directory with experiment name
        checkpoint_path = os.path.join(checkpoint_path, args.exp_name)
        try:
            os.makedirs(checkpoint_path)
        except OSError:
            pass
    


    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print('initializing environments')
    #Andy: Add option to turn off environment normalization
    print('Normalize Env:', args.normalize_env)
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, capture_video=args.capture_video,
                         env_kwargs=args.env_kwargs, normalize=args.normalize_env,
                         **args.aux_wrapper_kwargs)

    loaded_model = False
    # print(args.cont)
    print('initializing model')
    print(args.save_interval)
    
    #Andy: for continuing an experiment, args.cont is True
    if args.cont:
        loaded_model = True
        actor_critic, obs_rms = torch.load(save_path)
    
    if not loaded_model:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base=args.nn_base,
            base_kwargs={'recurrent': args.recurrent_policy, 
                         **args.nn_base_kwargs})
        
        #Andy: if cloning parameters, do it here. We are assuming
        #that the target cloning network has the appropriate parameter
        #sizes
        if args.clone_parameter_experiment:
            clone_args = args.clone_args
            clone_actor_critic, obs_rms = torch.load(clone_args['clone_path'])
            clone_layers = clone_args['clone_layers']
            freeze_layers = clone_args['freeze']
            
            clone_params = list(clone_actor_critic.parameters())
            actor_critic_params = list(actor_critic.parameters())
            
            if type(clone_layers) == int:
                clone_layers = range(clone_layers)
            if type(freeze_layers) == bool:
                freeze_layers = [freeze_layers]*len(clone_layers)
            for i, layer in enumerate(clone_layers):
                freeze = freeze_layers[i]
                
                actor_critic_params[layer].data.copy_(clone_params[layer].data)
                if freeze:
                    actor_critic_params[layer].requires_grad = False
        
        actor_critic.to(device)

    print('initializing algo')
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        if args.use_new_aux:
            agent = PPOAux(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                args.auxiliary_loss_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
                remove_actor_grads_on_shared=args.remove_actor_grads_on_shared)   
        else:
            agent = algo.PPO(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                args.auxiliary_loss_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
                remove_actor_grads_on_shared=args.remove_actor_grads_on_shared)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    print('initializing storage')
    if args.use_new_aux:
        rollouts = RolloutStorageAux(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space,
                                actor_critic.recurrent_hidden_state_size,
                                actor_critic.auxiliary_output_sizes, #note sizes not size here
                                auxiliary_truth_sizes=args.auxiliary_truth_sizes)
    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space,
                                actor_critic.recurrent_hidden_state_size,
                                actor_critic.auxiliary_output_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    #Andy: add global step
    if args.cont:
        global_step = int(obs_rms.count)
    else:
        global_step = 0
    # print(global_step)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        #1. Collect args.num_steps * args.num_processes number of experience steps
        #from the environment
        for step in range(args.num_steps):
            #Andy: add global step
            global_step += 1 * args.num_processes
            # Sample actions
            
            #Andy: Update to use outputs dict
            with torch.no_grad():
                # value, action, action_log_prob, recurrent_hidden_states, auxiliary_preds = \
                # actor_critic.act(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                #     rollouts.masks[step])
                outputs = actor_critic.act(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                action = outputs['action']
                value = outputs['value']
                action_log_prob = outputs['action_log_probs']
                recurrent_hidden_states = outputs['rnn_hxs']
                auxiliary_preds = outputs['auxiliary_preds']



            # Obser reward and next obs
            # obs, reward, done, infos = envs.step(action)
            obs, reward, done, infos = envs.step(action)
            
            if args.use_new_aux:
                # !! New auxiliary code separating out auxiliary tasks into list
                # Note that we will have an error "list index out of range" if 
                #  there is a mismatch between number of truths and preds
                auxiliary_truths = [[] for i in range(len(actor_critic.auxiliary_output_sizes))]
                for info in infos:
                    if 'auxiliary' in info and len(info['auxiliary']) > 0:
                        for i, aux in enumerate(info['auxiliary']):
                            auxiliary_truths[i].append(aux)
                if len(auxiliary_truths) > 0:
                    auxiliary_truths = [torch.tensor(np.vstack(aux)) for aux in auxiliary_truths]

            else:
                # !! Old auxiliary code needed to run old training model
                auxiliary_truths = []
                for info in infos:
                    if 'auxiliary' in info:
                        if len(info['auxiliary'] > 0):
                            auxiliary_truths.append(info['auxiliary'])
                if len(auxiliary_truths) > 0:
                    auxiliary_truths = torch.tensor(np.vstack(auxiliary_truths))
                else:
                    auxiliary_truths = None
            
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    print(f'global_step={global_step}')
                    # Andy: add tensorboard writing episode returns
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], 
                        global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], 
                        global_step)



            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks,
                            auxiliary_preds, auxiliary_truths)

        #2. Compute rewards and update parameters with policy improvement
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        
        if args.algo == 'ppo':
            value_loss, action_loss, dist_entropy, approx_kl, clipfracs, \
                auxiliary_loss = agent.update(rollouts)

        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        #Andy: add tensorboard data tracking
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss, global_step)
        writer.add_scalar("losses/policy_loss", action_loss, global_step)
        writer.add_scalar("losses/auxiliary_loss", auxiliary_loss, global_step)
        writer.add_scalar("losses/entropy", dist_entropy, global_step)

        if args.algo == 'ppo':
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start)), global_step)

        #Attempt to manually log any videos to W&B if they exist
        if args.capture_video and 'video' in os.listdir():
            upload_videos()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":

            # Andy: change save path to use more args
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], save_path)

        # Andy: if checkpointing, save every interval-th episode
        # Note that 0th update is actually the 1st update, because this comes after update code
        if args.checkpoint_interval > 0 and (j % args.checkpoint_interval == 0 or j == num_updates - 1):
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(checkpoint_path, str(j) + '.pt'))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))



        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

    #Finally log all remaining videos
    if args.track and args.capture_video and 'video' in os.listdir():
        time.sleep(5)
        upload_videos()



    if args.config_file_name is not None:
        write_latest_exp_complete(args.config_file_name)
        print('Experiment completed, experiment log updated')


if __name__ == "__main__":
    main()
