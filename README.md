# pytorch-a2c-ppo-acktr

## Fork

This is Andrew Liu's fork of pytorch-a2c-ppo-acktr-gail

Here I will be making some slight adjustments to code to make it nicer to track experiments with

### Changes from Original

In each of the files, you can search "Andy:" for places where the original code was changed.

* main.py: add Tensorboard and log every update
  * also add optionality to using wandb (stopped working on CHPC)
  * add optionality to save checkpoints
* a2c_ppo/arguments.py: add flags for tracking and video capture and checkpointing and using specific kwargs
* a2c_ppo/algo/ppo.py: add calculations for approx kl divergence and clipfracs
* a2c_ppo/envs.py: add video capture wrapper to environments, ability to use env_kwargs
* a2c_ppo/model.py: added FlexBase for changing how many layers are shared between actor and critic

* evaluate.py: adding code to return all seen obs and actions during evaluation, as well as hidden states so that we could potentitally map out hidden state trajectories. Adding code to take an optional callback to gather additional data from environment, and option to select how many episodes
  * added ability to use to record video and verbosity for printing individual episode results

### Second Auxiliary Task Training Methods



### Research Work

In the write_and_test folder, I am working on code to either set up experiments, make plots, or test code. These are mostly handled in Jupyter notebooks

## Code Flow Notes

To generate a model, main.py creates a 

Policy (model.py)
* Uses MLPBase if observation is 1-dim, CNNBase if 3-dim
* self.base = MLPBase: base is the network used in Policy
* base outputs a critic value, actor hidden activatios, and rnn hidden states
* self.dist converts actor hidden activations to action output. For discrete actions, it is Categorical (from distributions.py) which adds a hidden -> n linear output and does categorical distribution
* save rollouts during episode to self.rollouts, which is an instance of RolloutsStorage

MLPBase
* Two separate networks are created for actor and critic
* Tanh activations used
* actor and critic have layers input -> hidden -> hidden (2 hidden layers), default to 64 hidden units
* critic has built in final layer hidden -> 1, so outputs value
* if self.recurrent, adds in a GRU layer input -> hidden, and changes the inputs of actor and critic to hidden

RolloutsStorage
* For recurrent policy, saves subsequent hidden states detached, so no rollback through time during updates



## Update (April 12th, 2021)

PPO is great, but [Soft Actor Critic](https://arxiv.org/abs/1812.05905) can be better for many continuous control tasks. Please check out [my new RL](http://github.com/ikostrikov/jax-rl) repository in jax.

## Please use hyper parameters from this readme. With other hyper parameters things might not work (it's RL after all)!

This is a PyTorch implementation of
* Advantage Actor Critic (A2C), a synchronous deterministic version of [A3C](https://arxiv.org/pdf/1602.01783v1.pdf)
* Proximal Policy Optimization [PPO](https://arxiv.org/pdf/1707.06347.pdf)
* Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation [ACKTR](https://arxiv.org/abs/1708.05144)
* Generative Adversarial Imitation Learning [GAIL](https://arxiv.org/abs/1606.03476)

Also see the OpenAI posts: [A2C/ACKTR](https://blog.openai.com/baselines-acktr-a2c/) and [PPO](https://blog.openai.com/openai-baselines-ppo/) for more information.

This implementation is inspired by the OpenAI baselines for [A2C](https://github.com/openai/baselines/tree/master/baselines/a2c), [ACKTR](https://github.com/openai/baselines/tree/master/baselines/acktr) and [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1). It uses the same hyper parameters and the model since they were well tuned for Atari games.

Please use this bibtex if you want to cite this repository in your publications:

    @misc{pytorchrl,
      author = {Kostrikov, Ilya},
      title = {PyTorch Implementations of Reinforcement Learning Algorithms},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail}},
    }

## Supported (and tested) environments (via [OpenAI Gym](https://gym.openai.com))
* [Atari Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
* [MuJoCo](http://mujoco.org)
* [PyBullet](http://pybullet.org) (including Racecar, Minitaur and Kuka)
* [DeepMind Control Suite](https://github.com/deepmind/dm_control) (via [dm_control2gym](https://github.com/martinseilair/dm_control2gym))

I highly recommend PyBullet as a free open source alternative to MuJoCo for continuous control tasks.

All environments are operated using exactly the same Gym interface. See their documentations for a comprehensive list.

To use the DeepMind Control Suite environments, set the flag `--env-name dm.<domain_name>.<task_name>`, where `domain_name` and `task_name` are the name of a domain (e.g. `hopper`) and a task within that domain (e.g. `stand`) from the DeepMind Control Suite. Refer to their repo and their [tech report](https://arxiv.org/abs/1801.00690) for a full list of available domains and tasks. Other than setting the task, the API for interacting with the environment is exactly the same as for all the Gym environments thanks to [dm_control2gym](https://github.com/martinseilair/dm_control2gym).

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [Stable baselines3](https://github.com/DLR-RM/stable-baselines3)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Other requirements
pip install -r requirements.txt
```

## Contributions

Contributions are very welcome. If you know how to make this code better, please open an issue. If you want to submit a pull request, please open an issue first. Also see a todo list below.

Also I'm searching for volunteers to run all experiments on Atari and MuJoCo (with multiple random seeds).

## Disclaimer

It's extremely difficult to reproduce results for Reinforcement Learning methods. See ["Deep Reinforcement Learning that Matters"](https://arxiv.org/abs/1709.06560) for more information. I tried to reproduce OpenAI results as closely as possible. However, majors differences in performance can be caused even by minor differences in TensorFlow and PyTorch libraries.

### TODO
* Improve this README file. Rearrange images.
* Improve performance of KFAC, see kfac.py for more information
* Run evaluation for all games and algorithms

## Visualization

In order to visualize the results use ```visualize.ipynb```.


## Training

### Atari
#### A2C

```bash
python main.py --env-name "PongNoFrameskip-v4"
```

#### PPO

```bash
python main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
```

#### ACKTR

```bash
python main.py --env-name "PongNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20
```

### MuJoCo

Please always try to use  ```--use-proper-time-limits``` flag. It properly handles partial trajectories (see https://github.com/sfujim/TD3/blob/master/main.py#L123).

#### A2C

```bash
python main.py --env-name "Reacher-v2" --num-env-steps 1000000
```

#### PPO

```bash
python main.py --env-name "Reacher-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits
```

#### ACKTR

ACKTR requires some modifications to be made specifically for MuJoCo. But at the moment, I want to keep this code as unified as possible. Thus, I'm going for better ways to integrate it into the codebase.

## Enjoy

### Atari

```bash
python enjoy.py --load-dir trained_models/a2c --env-name "PongNoFrameskip-v4"
```

### MuJoCo

```bash
python enjoy.py --load-dir trained_models/ppo --env-name "Reacher-v2"
```

## Results

### A2C

![BreakoutNoFrameskip-v4](imgs/a2c_breakout.png)

![SeaquestNoFrameskip-v4](imgs/a2c_seaquest.png)

![QbertNoFrameskip-v4](imgs/a2c_qbert.png)

![beamriderNoFrameskip-v4](imgs/a2c_beamrider.png)

### PPO


![BreakoutNoFrameskip-v4](imgs/ppo_halfcheetah.png)

![SeaquestNoFrameskip-v4](imgs/ppo_hopper.png)

![QbertNoFrameskip-v4](imgs/ppo_reacher.png)

![beamriderNoFrameskip-v4](imgs/ppo_walker.png)


### ACKTR

![BreakoutNoFrameskip-v4](imgs/acktr_breakout.png)

![SeaquestNoFrameskip-v4](imgs/acktr_seaquest.png)

![QbertNoFrameskip-v4](imgs/acktr_qbert.png)

![beamriderNoFrameskip-v4](imgs/acktr_beamrider.png)
