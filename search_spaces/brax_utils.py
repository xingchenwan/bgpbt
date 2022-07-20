import functools
import pathlib

import brax
import logging
from brax import envs
from brax.envs import to_torch
from torch import nn
import torch
import numpy as np
import random
import gym
from custom_brax_train import ppo_torch as ppo


def get_default_config():
    """Get the default value of the configurations -- used currently for distillation"""
    config = {
        'logdir': 'ppo_torch',
        'idx': 0,
        'seed': 0,
        'num_timesteps': int(3e7),
        'episode_length': 1000,
        'discounting': 0.97,
        'learning_rate': 0.0003,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'batch_size': 1024,
        'num_minibatches': 32,
        'num_update_epochs': 4,
        'reward_scaling': 10,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
        'policy_hidden_layer_sizes': "32, 32, 32, 32",
        'policy_activation': 'silu',
        'v_hidden_layer_sizes': '256,256, 256, 256, 256',
        'v_activation': 'silu',
        'num_envs': 2048,
        'eval_every': int(5e5),
        'env_name': 'ant'
    }
    return config


def get_brax_trainer(params: dict, load_path: str = None, mismatch_policy: str = 'ignore', alg_name='ppo'):
    """
    Helper function to get a brax trainer.
    param: the dict containing the hyperparameters
    save_path: the checkpoint path. If this is specified, this def will try to restore the trainer state from the
        save_path. Otherwise a new trainer will be created
    """
    assert mismatch_policy in ['ignore', 'strict']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'elu': nn.ELU(),
        'silu': nn.SiLU(),
        'hardswish': nn.Hardswish(),
    }

    # re-implementation of the same content in main_ppo_torch but within a unified python interface (without calling
    # bash scrhipts which can be problematic for multi-phase training)
    if 'logdir' not in params.keys():
        params['logdir'] = f'{alg_name}_torch'
    if 'idx' not in params.keys():
        params['idx'] = 0
    if 'seed' not in params.keys():
        params['seed'] = 0
    if 'num_timesteps' not in params.keys():
        params['num_timesteps'] = int(3e7)
    if 'episode_length' not in params.keys():
        params['episode_length'] = 1000
    if 'discounting' not in params.keys():
        params['discounting'] = 0.97
    if 'learning_rate' not in params.keys():
        params['learning_rate'] = 0.0003
    if 'entropy_cost' not in params.keys():
        params['entropy_cost'] = 0.01
    if 'unroll_length' not in params.keys():
        params['unroll_length'] = 5
    if 'batch_size' not in params.keys():
        params['batch_size'] = 1024
    if 'num_minibatches' not in params.keys():
        params['num_minibatches'] = 32
    if 'num_update_epochs' not in params.keys():
        params['num_update_epochs'] = 4
    if 'reward_scaling' not in params.keys():
        params['reward_scaling'] = 10.
    if 'lambda_' not in params.keys():
        params['lambda_'] = .95

    # only difference between PPO and A2C is that A2C does not have the PPO epsilon param
    if alg_name == 'ppo' and ('ppo_epsilon' not in params.keys()):
        params['ppo_epsilon'] = 0.3
    elif alg_name == 'a2c' and 'ppo_epsilon' in params.keys():
        del params['ppo_epsilon']

    if 'policy_hidden_layer_sizes' not in params.keys():
        params['policy_hidden_layer_sizes'] = "32,32,32,32"
    if 'policy_activation' not in params.keys():
        params['policy_activation'] = "silu"
    if 'v_hidden_layer_sizes' not in params.keys():
        params['v_hidden_layer_sizes'] = "256,256,256,256,256"
    if 'v_activation' not in params.keys():
        params['v_activation'] = 'silu'
    if 'num_envs' not in params.keys():
        params['num_envs'] = 2048
    if 'eval_every' not in params.keys():
        params['eval_every'] = int(5e5)
    if 'env_name' not in params.keys():
        params['env_name'] = 'ant'
    if 'anneal_lr' not in params.keys():
        params['anneal_lr'] = False

    if isinstance(params['logdir'], str):  # parse into a string
        logdir = pathlib.Path(params['logdir']).expanduser()
    else:
        logdir = params['logdir']
    logdir.mkdir(parents=True, exist_ok=True)

    # Seeding
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    gym_name = f'brax-{params["env_name"]}-v0'
    if gym_name not in gym.envs.registry.env_specs:
        entry_point = functools.partial(
            envs.create_gym_env, env_name=params["env_name"])
        gym.register(gym_name, entry_point=entry_point)
    env = gym.make(
        gym_name, batch_size=params["num_envs"], episode_length=params["episode_length"])
    env = to_torch.JaxToTorchWrapper(env, device=device)
    # Warm-start
    env.reset()
    action = torch.zeros(env.action_space.shape).to(device)
    env.step(action)

    results_save_path = logdir / \
        f'progress_idx{params["idx"]}_seed{params["seed"]}.json'

    agent_class = ppo.Agent
    trainer_class = ppo.PPOTrainer
    logging.info(f'Getting trainer = {alg_name}')
    # Create agent
    agent = agent_class(
        obs_dim=env.observation_space.shape[-1],
        act_dim=env.action_space.shape[-1],
        policy_hidden_layer_sizes=params['policy_hidden_layer_sizes'],
        policy_activation=activations[params['policy_activation']],
        v_hidden_layer_sizes=params['v_hidden_layer_sizes'],
        value_activation=activations[params['v_activation']],
        entropy_cost=params['entropy_cost'],
        discounting=params['discounting'],
        reward_scaling=params['reward_scaling'],
        lambda_=params['lambda_'],
        ppo_epsilon=params['ppo_epsilon'] if alg_name == 'ppo' else None,
        unroll_length=params['unroll_length'],
        batch_size=params['batch_size'],
        num_minibatches=params['num_minibatches'],
        num_update_epochs=params['num_update_epochs'],
        device=device,

    )
    agent = torch.jit.script(agent.to(device))

    trainer = trainer_class(
        env=env,
        agent=agent,
        learning_rate=params['learning_rate'],
        save_path=results_save_path,
        episode_length=params['episode_length'],
        eval_every=params['eval_every'],
        schedule_lr=params['anneal_lr'])
    if load_path is not None:
        try:
            trainer.load_checkpoint(load_path)
            trainer.update_training_hyperparameters(**params)
            logging.info(f'Trainer state restored from {load_path}')
        except Exception as e:
            if mismatch_policy == 'ignore':
                logging.warning(
                    f'Failed to restore trainer state from {load_path} with exception: {e}. A new trainer has been created.')
            else:
                raise Exception(e)
    return trainer
