# SHAC

This repository contains the implementation for the paper [Practical Critic Gradient based Actor Critic for On-Policy Reinforcement Learning](https://openreview.net/forum?id=ddl_4qQKFmY) (L4DC 2023).

In this paper, we present a different class of on-policy algorithms based on SARSA, which estimate the policy gradient using the critic-action gradients. We show that they are better suited than existing baselines (like PPO) especially when using highly parallelizable simulators. We observe that the critic gradient based on-policy method (CGAC) consistently achieves higher episode returns. Furthermore, in environments with high dimensional action space, CGAC also trains much faster (in wall-clock time) than the corresponding baselines.

## Installation

- `git clone https://github.com/NVlabs/DiffRL.git --recursive`

#### Prerequisites

- In the project folder, create a virtual environment in Anaconda:

  ```
  conda env create -f diffrl_conda.yml
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

  Install Isaacgym with instructions from their [github repo]().

## Training

Running the following commands in `examples` folder allows to train Ant with SHAC.
```
python train_shac.py --cfg ./cfg/shac/ant.yaml --logdir ./logs/Ant/shac
```
The results might slightly differ from the paper due to the randomness of the cuda and different Operating System/GPU/Python versions. The experiments in this paper were done on CentOS 7 with a NVIDIA GTX 2080 gpu.

#### SHAC (Our Method)

For example, running the following commands in `examples` folder allows to train Ant and SNU Humanoid (Humanoid MTU in the paper) environments with SHAC respectively for 5 individual seeds.

```
python train_script.py --env Ant --algo shac --num-seeds 5
```

```
python train_script.py --env SNUHumanoid --algo shac --num-seeds 5
```

#### Baseline Algorithms

For example, running the following commands in `examples` folder allows to train Ant environment with PPO and SAC implemented in RL_games for 5 individual seeds,

```
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