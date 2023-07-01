from a2c_ppo_acktr.arguments import get_args
from scheduler import convert_config_to_command
from a2c_ppo_acktr.algo.ppo import PPOAux
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.envs import make_vec_envs
import torch

args = get_args()
print(args)

def run_config():
    device = torch.device('cpu')
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                            args.gamma, args.log_dir, device, False, capture_video=args.capture_video,
                            env_kwargs=args.env_kwargs, normalize=args.normalize_env,
                            **args.aux_wrapper_kwargs)
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base=args.nn_base,
        base_kwargs={'recurrent': args.recurrent_policy, 
                        **args.nn_base_kwargs})
    
    return envs, actor_critic