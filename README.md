# SHAC

This repository contains the implementation for the paper [Practical Critic Gradient based Actor Critic for On-Policy Reinforcement Learning](https://openreview.net/forum?id=ddl_4qQKFmY) (L4DC 2023).

In this paper, we present a different class of on-policy algorithms based on SARSA, which estimate the policy gradient using the critic-action gradients. We show that they are better suited than existing baselines (like PPO) especially when using highly parallelizable simulators. We observe that the critic gradient based on-policy method (CGAC) consistently achieves higher episode returns. Furthermore, in environments with high dimensional action space, CGAC also trains much faster (in wall-clock time) than the corresponding baselines.

## Installation

- `git clone https://github.com/NVlabs/DiffRL.git --recursive`

#### Prerequisites

- In the project folder, create a virtual environment in Anaconda:

  ```
  conda env create -f cgac.yml
  conda activate cgac
  ```

- dflex

  ```
  cd dflex
  pip install -e .
  ```

- rl_games, forked from [rl-games](https://github.com/Denys88/rl_games) (used for PPO and SAC training):

  ````
  cd externals/rl_games
  pip install -e .
  ````

- Install an older version of protobuf required for TensorboardX:
  ````
  pip install protobuf==3.20.0
  ````

  Install Isaacgym and Isaacgymenvs with instructions from their [github repo](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs).

## Training

The results might slightly differ from the paper due to the randomness of the cuda and different Operating System/GPU/Python versions. The experiments in this paper were done on CentOS 7 with a NVIDIA GTX 2080 gpu.

#### CGAC (Our Method)

Run the following commands with the appropriate flags to train a model corresponding to each environment from inside the `cgac` folder.

```
python main.py --env-name AntEnv --cuda --num_actors 4096 --batch_size_update 4096 --critic_hidden 1024 1024 256 --actor_hidden 256 256 --critic_act elu --actor_act elu --num_critic_updates 2 --grad_norm 20 --final_targ_ent_coeff 3.5 --no_automatic_entropy_tuning False --alpha_schedule constant --alpha 1e-1 --final_lr 1e-5 --clip_actor_gn --seed 1 --final

python main.py --env-name HumanoidEnv --cuda --num_actors 4096 --batch_size_update 4096 --critic_hidden 1024 1024 256 --actor_hidden 256 256 --critic_act elu --actor_act elu --num_critic_updates 2 --grad_norm 20 --final_targ_ent_coeff 7.5 --seed 0 --final

python main.py --env-name SNUHumanoidEnv --cuda --num_actors 4096 --batch_size_update 4096 --critic_hidden 512 512 256 --actor_hidden 256 256 --critic_act elu --actor_act elu --num_critic_updates 6 --grad_norm 20 --final_targ_ent_coeff 3.5 --seed 0 --val_horizon 8 --final

python main.py --env-name AllegroHand --cuda --num_actors 16384 --batch_size_update 32768 --critic_hidden 1024 512 256 --actor_hidden 512 256 --critic_act elu --actor_act elu --num_critic_updates 2 --final_targ_ent_coeff 5 --alpha 0.2 --init_targ_ent_coeff 0.1 --val_horizon 8 --tau_value 0.05 --tau_policy 1.0 --final
```

#### Baseline Algorithms

In order to train the baselines for the Dflex environments (Ant, Humanoid, SNUHumanoid), simply run the following commands in the `examples` folder while switching the env-name of choice.

```
python train_script.py --env Ant --algo sac --num-seeds 5
python train_script.py --env Ant --algo ppo --num-seeds 5
```

### Acknowledgments

The dflex env and code for testing the baselines was borrowed from the [DiffRL](https://github.com/NVlabs/DiffRL.git) repo. We also build on code from (https://github.com/pranz24/pytorch-soft-actor-critic) for various aspects of the model. We thank the authors of these repos for making their code public.

## Citation

If you find our paper or code is useful, please consider citing:
```
  @inproceedings{
  gurumurthy2023practical,
  title={Practical Critic Gradient based Actor Critic for On-Policy Reinforcement Learning},
  author={Swaminathan Gurumurthy and Zachary Manchester and J Zico Kolter},
  booktitle={5th Annual Learning for Dynamics {\&} Control Conference},
  year={2023},
  url={https://openreview.net/forum?id=ddl_4qQKFmY}
  }
```