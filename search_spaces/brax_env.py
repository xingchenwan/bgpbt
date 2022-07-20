import logging
import sys

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import torch.multiprocessing.pool

from custom_brax_train.default_configs import PPO_DEFAULT_CONFIGS
from copy import deepcopy
import numpy as np
import subprocess
import time
import json
from pathlib import Path
from typing import List
from definitions import ROOT_DIR
from search_spaces.base import SearchSpace
from search_spaces.brax_utils import get_brax_trainer


class Brax(SearchSpace):

    def __init__(self, env_name: str,
                 log_dir: str,
                 alg_name: str = 'ppo',
                 seed: int = 0,
                 max_parallel: int = 4,
                 do_nas=True,
                 do_hpo=True,
                 choose_spectral_norm: bool = True,
                 use_same_arch_policy_q: bool = False,
                 failed_run_penalty: float = -5000,
                 use_cmd: bool = False,
                 use_categorical: bool = False,
                 smaller_network: bool = False,
                 dry_run=False):
        """
        env_name: 
        """
        super(Brax, self).__init__()
        assert env_name in ['ant', 'humanoid', 'fetch',
                            'grasp', 'halfcheetah', 'ur5e', 'reacher', 'hopper']
        # todo: support more RL algorithms
        assert alg_name in ['ppo', ]

        assert 0 < max_parallel
        # whether use pytorch env
        self.use_cmd = use_cmd
        self.max_parallel = max_parallel
        self.env_name = env_name
        self.alg_name = alg_name
        self.failed_run_penalty = failed_run_penalty
        # self.seeds = np.arange(1000)
        self.seed = seed
        self.use_categorical = use_categorical
        if log_dir[0] != '/':
            self.log_dir = Path(log_dir).absolute()
        else:
            self.log_dir = Path(log_dir)
        self.smaller_network = smaller_network

        # some boundaries -- initialised here as they themselves are used to specify bounds of other hyperparams.
        self.min_n_hidden_layers = 1
        self.max_n_hidden_layers = 5 if not self.smaller_network else 2
        self.dry_run = dry_run
        self.config_space = self.get_configspace(
            do_nas, do_hpo, choose_spectral_norm, use_same_arch_policy_q)
        if alg_name == 'ppo' and env_name in PPO_DEFAULT_CONFIGS.keys():
            # for PPO, Brax provides a set of existing hyperparams that we may load (and overwrite)
            self.default_env_args = PPO_DEFAULT_CONFIGS[env_name]
            self.default_env_args['env_name'] = self.env_name
        else:
            # fallback values if default args are not provided by Google (example: hopper)
            self.default_env_args = {'num_timesteps': 20_000_000,
                                     'log_frequency': 20,
                                     'reward_scaling': 10, 'episode_length': 1000,
                                     'normalize_observations': True, 'action_repeat': 1,
                                     'unroll_length': 5, "num_minibatches": 32,
                                     "num_update_epochs": 4, "discounting": 0.97, "learning_rate": 3e-4,
                                     "entropy_cost": 1e-2, "num_envs": 2048,
                                     'batch_size': 1024,
                                     'env_name': self.env_name}

        if self.dry_run:
            self.default_env_args['num_timesteps'] = 1_000_000

    def get_configspace(self, do_nas: bool = None,
                        do_hpo: bool = None,
                        choose_spectral_norm: bool = True,
                        use_same_arch_policy_q: bool = False):
        """

        :param do_nas: whether to do neural architecture search
        :param do_hpo: whether to do hyperparameter optimisation
        :param choose_spectral_norm: whether select between the MLP/MLP with spectral norm. If False, we will simply
            use MLP without spectral norm as the default architecture.
        :param use_same_arch_policy_q: whether constrain the policy and q networks such that they have the same
            architecture

        :return:
        """
        cs = CS.ConfigurationSpace(
            seed=self.seed)  # configspace seems buggy in testing some of the inactive values
        # the nas hyperparameter start with NAS_ prefix!
        if do_nas:
            nets = ['policy'] if use_same_arch_policy_q else ['policy', 'q']
            for net in nets:  # for sac, tune policy and action networks separately
                # MLP or spectral norm MLP
                if choose_spectral_norm:
                    use_spectral_norm = CSH.UniformIntegerHyperparameter(f'NAS_{net}_use_spectral_norm', 0, 1,
                                                                         default_value=1)
                    cs.add_hyperparameter(use_spectral_norm)
                if self.use_categorical:
                    num_hidden_layers = CSH.CategoricalHyperparameter(f'NAS_{net}_num_layers', choices=list(
                        range(self.min_n_hidden_layers, self.max_n_hidden_layers+1)))
                else:
                    num_hidden_layers = CSH.UniformIntegerHyperparameter(f'NAS_{net}_num_layers', self.min_n_hidden_layers,
                                                                         self.max_n_hidden_layers, default_value=2)
                if self.smaller_network:
                    network_width = CSH.CategoricalHyperparameter(f'NAS_{net}_log2_width', choices=[5, 6]) \
                        if self.use_categorical \
                        else CSH.UniformIntegerHyperparameter(f'NAS_{net}_log2_width', 5, 6, default_value=5)
                else:
                    network_width = CSH.CategoricalHyperparameter(f'NAS_{net}_log2_width', choices=[5, 6, 7, 8]) \
                        if self.use_categorical\
                        else CSH.UniformIntegerHyperparameter(f'NAS_{net}_log2_width', 5, 8, default_value=5)
                cs.add_hyperparameter(num_hidden_layers)
                cs.add_hyperparameter(network_width)

        if do_hpo:
            if self.use_categorical:
                hyperparams = [CSH.UniformFloatHyperparameter('log10_lr', -4, -3, default_value=np.log10(3e-4)),
                               CSH.UniformFloatHyperparameter(
                    'discounting', 0.9, 0.9999, default_value=0.97, log=True),
                    CSH.UniformFloatHyperparameter(
                    'log10_entropy_cost', -6, -1, default_value=-2),
                    CSH.CategoricalHyperparameter('unroll_length', choices=list(
                        range(5, 16)), ),  # 5 - 15 inclusive
                    CSH.CategoricalHyperparameter(
                    'log2_batch_size', choices=list(range(6, 11))),  # 6 - 10 inclusive
                    CSH.CategoricalHyperparameter(
                    'num_update_epochs', choices=list(range(2, 17))),  # 2 - 16 inclusive
                    CSH.UniformFloatHyperparameter(
                    'reward_scaling', 0.05, 20, default_value=10, log=True),
                    CSH.UniformFloatHyperparameter(
                    'lambda_', 0.9, 1, default_value=0.95)
                ]
                hyperparams.append(CSH.UniformFloatHyperparameter(
                    'ppo_epsilon', 0.1, 0.4, default_value=0.2))
            else:
                hyperparams = [CSH.UniformFloatHyperparameter('log10_lr', -4, -3, default_value=np.log10(3e-4)),
                               CSH.UniformFloatHyperparameter(
                    'discounting', 0.9, 0.9999, default_value=0.97, log=True),
                    CSH.UniformFloatHyperparameter(
                    'log10_entropy_cost', -6, -1, default_value=-2),
                    CSH.UniformIntegerHyperparameter(
                    'unroll_length', 5, 15, default_value=5),
                    # large unroll can lead to OOM
                    CSH.UniformIntegerHyperparameter(
                    'log2_batch_size', 6, 10, default_value=10),
                    CSH.UniformIntegerHyperparameter(
                    'num_update_epochs', 2, 16, default_value=4, log=True),
                    CSH.UniformFloatHyperparameter(
                    'reward_scaling', 0.05, 20, default_value=10, log=True),
                    CSH.UniformFloatHyperparameter('lambda_', 0.9, 1, default_value=0.95), ]
                hyperparams.append(CSH.UniformFloatHyperparameter(
                    'ppo_epsilon', 0.1, 0.4, default_value=0.2))
            # this should always be true, hence commented out as a hyperparameter for now
            cs.add_hyperparameters(hyperparams)
        return cs

    def get_default_config_str(self, return_str=True, return_dict=False):
        assert self.alg_name == 'ppo'
        default_arg_dict = PPO_DEFAULT_CONFIGS[self.env_name]
        default_arg_dict['logdir'] = self.log_dir
        default_str = []
        for k, v in default_arg_dict.items():
            default_str += f' --{k}={v}'
        default_str = "".join(default_str)
        if return_str and not return_dict:
            return default_str
        if return_dict and not return_str:
            return default_arg_dict
        if return_str and return_dict:
            return default_str, default_arg_dict

    def _parse_config(self, config: CS.Configuration, return_str=True, return_dict=False, **kwargs):
        """Parse configuration object into a string to be passed as command line arguments when calling the RL training
        script"""
        config_dict = config.get_dictionary()
        dict_to_append = {}
        for k, v in config_dict.items():
            # some parameters are log-wrapped -- reverse them back.
            if 'log' in k:
                if 'log2' in k:
                    k_ = k.split('log2_')[1]
                    dict_to_append[k_] = 2 ** config[k]
                elif 'log10' in k:
                    k_ = k.split('log10_')[1]
                    dict_to_append[k_] = 10 ** config[k]
                else:
                    k_ = k.split('log')[1]
                    dict_to_append[k_] = np.exp(config[k])
                # snap integer type to integers via rounding
                if type(self.config_space.get_hyperparameter(k)) in [CSH.UniformIntegerHyperparameter,
                                                                     CSH.NormalIntegerHyperparameter]:
                    dict_to_append[k_] = int(np.round(dict_to_append[k_]))

        config_dict.update(dict_to_append)

        # merge with architecture parameters
        arch_args = self._parse_architecture(config_dict)

        arg_dict = deepcopy(self.default_env_args)
        for k, v in arg_dict.items():
            # replace with additional keyword arguments passed here (which should be the fixed hyperparameters)
            # if k in kwargs.keys() and kwargs[k] is not None: arg_dict[k] = kwargs[k]
            if k in config.keys() and config[k] is not None:
                arg_dict[k] = config[k]

        # also merge with architecture information
        arg_dict = {**arg_dict, **arch_args, **kwargs, 'logdir': self.log_dir}
        args_str = []
        for k, v in arg_dict.items():
            if isinstance(v, tuple):
                if len(v) == 1:
                    args_str += f' --{k}="{str(v[0])}"'
                else:
                    args_str += f' --{k}="{str(v)[1:-1]}"'
            else:
                args_str += f' --{k}={v}'
        args_str = "".join(args_str)
        if return_str and not return_dict:
            return args_str
        if return_dict and not return_str:
            return arg_dict
        if return_dict and return_str:
            return args_str, arg_dict

    def train_single(self, config: CS.Configuration,
                     exp_idx: int,
                     seed: int = None,
                     num_timesteps: int = None,
                     train_default: bool = False,
                     previous_trainer=None,
                     return_trainer=False,
                     **kwargs):
        """Train a single configuration from scratch. If doing batch BO or other HPO, use train_batch.
        train_batch also handles when there is only a single config
        n_timesteps: if True, this overrides the default timestep setting. This can be useful for a multi-fidelity
            setting where we would like to run a configuration for a smaller budget.
        """
        if seed is None:
            seed = self.seed
        if train_default:
            args_str, args_dict = self.get_default_config_str(
                return_str=True, return_dict=True)
        else:
            if num_timesteps is not None:
                args_str, args_dict = self._parse_config(config, num_timesteps=num_timesteps, return_str=True,
                                                         return_dict=True, **kwargs)
            else:
                args_str, args_dict = self._parse_config(
                    config, return_str=True, return_dict=True, **kwargs)

        if not self.use_cmd:
            # Do not use the command line style -- create new trainers within the pytrhon interface
            trainer = previous_trainer if previous_trainer is not None else get_brax_trainer(
                args_dict, alg_name=self.alg_name)
            if previous_trainer is not None:
                trainer.update_training_hyperparameters(**args_dict)
            trainer.run_single_phase(num_timesteps=num_timesteps)
            all_metrics = trainer.metrics
            trajectory = {
                'x': all_metrics['num_steps'],
                'y': all_metrics['eval/episode_reward']
            }
            if return_trainer:
                return trajectory, trainer
            else:
                return trajectory

        else:
            if previous_trainer is not None:
                logging.warning(
                    'Ignored previous_trainer argument is not supported in the cmd mode.')
            cmd = f'cd {ROOT_DIR} && {sys.executable} -u main_{self.alg_name}.py --logdir={self.log_dir} --seed={seed} {args_str} --idx={exp_idx}'
            print(f'Executing {cmd}')

            process = subprocess.Popen(cmd,
                                       shell=True,
                                       )

            while process.poll() is None:
                print('Waiting for the training process to finish')
                time.sleep(60)
            stats = json.load(
                open(f'{self.log_dir}/progress_idx{exp_idx}_seed{seed}.json'))
            try:
                assert 'eval/episode_reward' in stats.keys()
                assert 'num_steps' in stats.keys()
                # save the trajectory in case of need
                trajectory = {
                    'x': stats['num_steps'],
                    'y': stats['eval/episode_reward'],
                }
            except Exception as e:
                print(
                    f'Exception occured with message={e}. The current run has been skipped. Assigning crashed runs penelty of {self.failed_run_penalty}')
                trajectory = {
                    'x': np.linspace(0, self.default_env_args['num_timesteps'],
                                     self.default_env_args['log_frequency']).tolist(),
                    'y': [self.failed_run_penalty] * self.default_env_args['log_frequency']
                }
            return trajectory

    def create_or_update_trainer(self, config: CS.Configuration, num_timesteps: int = None, prev_trainer=None,
                                 path=None):

        if num_timesteps is not None:
            args = self._parse_config(
                config, return_dict=True, return_str=False, num_timesteps=num_timesteps)
        else:
            args = self._parse_config(
                config, return_dict=True, return_str=False, )
        if prev_trainer is None:
            trainer = get_brax_trainer(args, path, alg_name=self.alg_name)
        else:
            trainer = prev_trainer
        trainer.update_training_hyperparameters(**args)
        return trainer

    def eval_checkpoint(self, config, checkpoint_path: str):
        arg_dict = self._parse_config(
            config, return_str=False, return_dict=True)
        trainer = get_brax_trainer(
            arg_dict, load_path=checkpoint_path, mismatch_policy='strict', alg_name=self.alg_name)
        progress = trainer.evaluate()
        return progress

    def train_batch(self, configs: List[CS.Configuration],
                    exp_idx_start: int = 0,
                    seeds: List[int] = None,
                    nums_timesteps: List[int] = None,
                    train_default: bool = False,
                    previous_trainers=None,
                    checkpoint_paths: List[str] = None,
                    max_parallel: int = None,
                    anneal_lr: bool = False,
                    **kwargs):
        """
        Train a batch of agents simultaneously (# parallel run is decided by the max_parallel argument).

        configs: a list of N ConfigSpace configurations specifying the hyperparameters for each parallel agent.
        num_timesteps:: a list of N int: specify the numbers of timestpes each agent should run
        previous_trainer: if provided, all new agents will continue training from the identical previous_trainer (but
            with potentially different hyperparameters)
        checkpoint_paths: a list of N path strs: if specified, each new agent will load its state dict from the checkpoint
            path. Useful for Population-based training.
        """
        max_parallel = max_parallel if max_parallel is not None else self.max_parallel
        n_config = len(configs)
        if seeds is not None:
            assert len(seeds) == n_config
        else:
            seeds = [self.seed] * n_config

        # if we are just using the default brax hyperparameters ...
        if train_default:
            args_strs_dicts = np.array([self.get_default_config_str(return_str=True, return_dict=True)] * len(configs),
                                       dtype=object)
            args_strs = args_strs_dicts[:, 0].tolist()
            args_dicts = args_strs_dicts[:, 1].tolist()
        else:
            if nums_timesteps is not None:
                assert len(nums_timesteps) == n_config
                args_strs_dicts = []
                for i, config in enumerate(configs):
                    args_strs_dicts.append(
                        self._parse_config(config, return_str=True, return_dict=True, num_timesteps=nums_timesteps[i],
                                           **kwargs)
                    )
                args_strs_dicts = np.array(args_strs_dicts, dtype=object)
            else:
                args_strs_dicts = np.array(
                    [self._parse_config(
                        config, return_str=True, return_dict=True, **kwargs) for config in configs],
                    dtype=object)
            args_strs = args_strs_dicts[:, 0].tolist()
            args_dicts = args_strs_dicts[:, 1].tolist()

        if seeds is not None and len(seeds):
            for i, seed in enumerate(seeds):
                args_dicts[i]['seed'] = seed
        if anneal_lr:
            for i in range(len(args_dicts)):
                args_dicts[i]['anneal_lr'] = True

        n_chunks = n_config // max_parallel + \
            1 if n_config % max_parallel else n_config // max_parallel
        chunk_size = min(n_config, max_parallel)

        offset = 0
        trajectories = []
        for i_chunk in range(n_chunks):
            current_chunk_strs = args_strs[offset:offset + chunk_size]
            current_chunk_dicts = args_dicts[offset: offset + chunk_size]
            current_chunk_ckpt_paths = checkpoint_paths[
                offset:offset + chunk_size] if checkpoint_paths is not None else [
                None] * chunk_size
            current_chunk_trainers = previous_trainers[
                offset:offset + chunk_size] if previous_trainers is not None else [
                None] * chunk_size
            alg_names = [self.alg_name] * chunk_size
            # spawn parallel processes

            if len(current_chunk_strs) > 1:
                if previous_trainers is not None:
                    args_list = list(zip(alg_names, current_chunk_dicts,
                                     current_chunk_ckpt_paths, current_chunk_trainers))
                else:
                    args_list = list(
                        zip(alg_names, current_chunk_dicts, current_chunk_ckpt_paths))
                p = torch.multiprocessing.Pool(len(current_chunk_strs))
                trajectory = p.map(_train, args_list)
                p.close()
                p.join()
            else:
                if previous_trainers is not None:
                    trajectory = _train(
                        (self.alg_name, current_chunk_dicts[0], current_chunk_ckpt_paths[0], current_chunk_trainers[0]))
                else:
                    trajectory = _train(
                        (self.alg_name, current_chunk_dicts[0], current_chunk_ckpt_paths[0]))
                trajectory = [trajectory]

            trajectories += trajectory
            offset += chunk_size

        return trajectories

    def distill_batch(self, teacher_configs: List[CS.Configuration], student_configs: List[CS.Configuration],
                      checkpoint_paths: List[str],
                      seeds: List[int] = None,
                      train_nums_timesteps: List[int] = None,
                      distill_nums_timesteps: List[int] = None,
                      distill_total_num_timesteps: int = None,
                      max_parallel: int = None,
                      fixed_teacher_params: dict = None,
                      fixed_student_params: dict = None,
                      replace_teacher=False):

        if fixed_student_params is None:
            fixed_student_params = {}
        if fixed_teacher_params is None:
            fixed_teacher_params = {}

        # these must match the arguments in ./custom_brax_train/ppo_torch.py: distill
        DEFAULT_DISTILLATION_PARAMS = {
            'policy_reg_coef': 5.,  # FLOAT
            'value_reg_coef': 0.,  # FLOAT
            'rl_coef': 1.,
            'distill_num_epochs': 4,  # INT
            'num_timesteps': distill_nums_timesteps[0]
        }
        # these must match the arguments in ./custom_brax_train/ppo_torch.py: set_distillation_schedule
        # these need to be passed to distillation at the very beginning.
        DISTILLATION_SCHEDULE = {
            'total_timesteps': distill_total_num_timesteps,
            'distill_anneal_frac': 0.8,
            'distill_anneal_init': 1,
            'distill_anneal_final': 0.05
        }
        n_config = len(teacher_configs)
        assert len(student_configs) == n_config
        assert len(checkpoint_paths) == n_config
        if seeds is not None:
            assert len(seeds) == n_config
        else:
            seeds = [self.seed] * n_config
        if max_parallel is None:
            max_parallel = self.max_parallel

        # parse the teacher and student configs
        if train_nums_timesteps is not None:
            assert len(train_nums_timesteps) == n_config
            teacher_args_strs_dicts, student_args_strs_dicts = [], []
            for i in range(n_config):
                teacher_args_strs_dicts.append(
                    self._parse_config(teacher_configs[i], return_str=True, return_dict=True,
                                       num_timesteps=train_nums_timesteps[i], **fixed_teacher_params)
                )
                student_args_strs_dicts.append(
                    self._parse_config(student_configs[i], return_str=True, return_dict=True,
                                       num_timesteps=train_nums_timesteps[i], **fixed_student_params)
                )
            teacher_args_strs_dicts = np.array(
                teacher_args_strs_dicts, dtype=object)
            student_args_strs_dict = np.array(
                student_args_strs_dicts, dtype=object)
        else:
            teacher_args_strs_dicts = np.array(
                [self._parse_config(config, return_str=True, return_dict=True, **fixed_teacher_params) for config in
                 teacher_configs],
                dtype=object)
            student_args_strs_dict = np.array(
                [self._parse_config(config, return_str=True, return_dict=True, **fixed_student_params) for config in
                 student_configs],
                dtype=object)

        teacher_arg_dicts = teacher_args_strs_dicts[:, 1].tolist()
        student_args_dicts = student_args_strs_dict[:, 1].tolist()

        if seeds is not None and len(seeds):
            for i, seed in enumerate(seeds):
                teacher_arg_dicts[i]['seed'] = seed
                student_args_dicts[i]['seed'] = seed

        n_chunks = n_config // max_parallel + \
            1 if n_config % max_parallel else n_config // max_parallel
        chunk_size = min(n_config, max_parallel)

        offset = 0
        trajectories = []
        for i_chunk in range(n_chunks):
            current_chunk_teacher_dicts = teacher_arg_dicts[offset: offset + chunk_size]
            current_chunk_student_dicts = student_args_dicts[offset: offset + chunk_size]
            current_chunk_ckpt_paths = checkpoint_paths[offset:offset + chunk_size]
            # spawn parallel processes

            args_list = list(zip([self.alg_name] * len(current_chunk_teacher_dicts),
                                 current_chunk_teacher_dicts,
                                 current_chunk_student_dicts,
                                 current_chunk_ckpt_paths,
                                 [DEFAULT_DISTILLATION_PARAMS] *
                                 len(current_chunk_teacher_dicts),
                                 [replace_teacher] *
                                 len(current_chunk_teacher_dicts),
                                 [DISTILLATION_SCHEDULE] *
                                 len(current_chunk_teacher_dicts)
                                 ),
                             )
            p = torch.multiprocessing.Pool(len(current_chunk_teacher_dicts))
            trajectory = p.map(_distill, args_list)
            p.close()
            p.join()
            trajectories += trajectory
            offset += chunk_size

        return trajectories

    def _parse_architecture(self, config: dict) -> dict:
        """Parse a Configuration into a valid tuple of ints e.g. (256, 128, 64, 256) that represent a particular
        MLP architecture
        Return a pair of tuples for both the policy network and the action network."""
        policy_net_spec = []
        q_net_spec = []
        # todo: make sure this is ok for other RL algorithms
        val_func_type = 'v'

        if 'NAS_q_num_layers' in config.keys():
            action_num_layers = config['NAS_q_num_layers']
            for i in range(1, action_num_layers + 1):
                q_net_spec.append(int(2 ** config['NAS_q_log2_width']))
        if 'NAS_policy_num_layers' in config.keys():
            policy_num_layers = config['NAS_policy_num_layers']
            for i in range(1, policy_num_layers + 1):
                policy_net_spec.append(
                    int(2 ** config['NAS_policy_log2_width']))
        policy_use_spectral_norm = True if 'NAS_policy_use_spectral_norm' in config.keys() and config[
            'NAS_policy_use_spectral_norm'] else False
        q_use_spectral_norm = True if 'NAS_q_use_spectral_norm' in config.keys() and config[
            'NAS_q_use_spectral_norm'] else False

        kwargs = {}
        if len(policy_net_spec):
            kwargs['policy_hidden_layer_sizes'] = list(policy_net_spec)
        if len(q_net_spec):
            kwargs[f'{val_func_type}_hidden_layer_sizes'] = list(q_net_spec)
        if 'NAS_policy_activation' in config.keys():
            kwargs['policy_activation'] = config['NAS_policy_activation']
            if 'NAS_q_activation' not in config.keys():  # when only one is specified, force q net to take same value
                kwargs[f'{val_func_type}_activation'] = config['policy_activation']
        else:
            kwargs[f'policy_activation'] = 'silu'

        if 'NAS_q_activation' in config.keys():
            kwargs[f'{val_func_type}_activation'] = config['NAS_q_activation']
        else:
            kwargs[f'{val_func_type}_activation'] = 'silu'

        kwargs[f'{val_func_type}_use_spectral_norm'] = q_use_spectral_norm
        kwargs['policy_use_spectral_norm'] = policy_use_spectral_norm

        # if there is only policy and no q spec: use the same architecture for both policy and q
        if len(policy_net_spec) and not len(q_net_spec):
            kwargs[f'{val_func_type}_hidden_layer_sizes'] = list(
                policy_net_spec)
            kwargs[f'{val_func_type}_use_spectral_norm'] = policy_use_spectral_norm

        # parse the HPO arguments, if they are present
        if 'lambda_' in config.keys():
            kwargs['lambda_'] = config['lambda_']
        if 'discounting' in config.keys():
            kwargs['discounting'] = config['discounting']
        if 'lr' in config.keys():
            kwargs['learning_rate'] = config['lr']  # note slight difference
        if 'batch_size' in config.keys():
            kwargs['batch_size'] = config['batch_size']
        if 'mini_batches' in config.keys():
            # note slight difference
            kwargs['num_minibatches'] = config['mini_batches']
        if 'entropy_cost' in config.keys():
            kwargs['entropy_cost'] = config['entropy_cost']
        if self.alg_name == 'ppo' and 'ppo_epsilon' in config.keys():
            kwargs['ppo_epsilon'] = config['ppo_epsilon']
        if 'num_update_epochs' in config.keys():
            kwargs['num_update_epochs'] = config['num_update_epochs']
        if 'reward_scaling' in config.keys():
            kwargs['reward_scaling'] = config['reward_scaling']
        if 'unroll_length' in config.keys():
            kwargs['unroll_length'] = config['unroll_length']
        return kwargs


def _train(args):
    """A quick wrapper to train a model and get its results. Placed in the top-level scope for multiprocessing."""
    torch.cuda.empty_cache()
    if len(args) == 3 or (len(args) == 4 and args[3] is None):
        alg_name, arg_dict, ckpt_path = args[0], args[1], args[2]
        trainer = get_brax_trainer(
            arg_dict, load_path=ckpt_path, alg_name=alg_name)
    elif len(args) == 4:
        alg_name, arg_dict, ckpt_path, trainer = args
    else:
        raise ValueError
    if len(trainer.students):
        logging.info('Student found -- replace teacher!')
        trainer.replace_teacher()
        trainer.students = []  # clear students for pure RL
    trainer.update_training_hyperparameters(**arg_dict)
    trainer.run(num_timesteps=arg_dict['num_timesteps'])
    trajectory = {
        'x': trainer.metrics['num_steps'],
        'y': trainer.metrics['eval/episode_reward']}
    if ckpt_path is not None:
        trainer.save_checkpoint(ckpt_path)
    del trainer
    return trajectory


def _distill(args):
    """similar to above but distill a model
    Similarly allows pausing and restoring distillations.
    """
    torch.cuda.empty_cache()
    alg_name, teacher_arg_dict, student_arg_dict, ckpt_path, distill_params, replace_teacher, distillation_schedules = args
    lr = student_arg_dict['learning_rate']
    student_arg_dict['learning_rate'] = 3e-4
    trainer = get_brax_trainer(
        teacher_arg_dict, load_path=ckpt_path, alg_name=alg_name)
    trainer.update_training_hyperparameters(**teacher_arg_dict)
    logging.info(f'Disillation params={distill_params}')
    if not len(trainer.students):
        logging.info('Starting distillation: attaching student!')
        trainer.set_distillation_schedule(**distillation_schedules)
        # todo: for now, use the default distillation hyperparams.
        trainer.attach_student(params=student_arg_dict)
    else:
        logging.info('Student found -- resuming distillation')
    trainer.distill(**distill_params)

    trajectory = {
        'x': trainer.metrics['num_steps'],
        'y': trainer.metrics['eval/episode_reward']
    }
    student_arg_dict['learning_rate'] = lr

    if replace_teacher:
        logging.info('Replacing teacher!')
        trainer.replace_teacher()
    trainer.save_checkpoint(ckpt_path)
    del trainer
    return trajectory
