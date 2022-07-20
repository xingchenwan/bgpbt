import numpy as np
import random
from search_spaces import Brax
from hpo import BGPBT
from hpo.casmo.bgpbt_arch import BGPBTArch
import argparse
import os
import logging
from definitions import ROOT_DIR
import torch

parser = argparse.ArgumentParser(
    description='Args to test performance sensitivity to hyperparameter choices on Brax tasks')
parser.add_argument('-e', '--env_name', type=str, default='ant',
                    choices=['ant', 'humanoid', 'fetch', 'grasp', 'halfcheetah', 'ur5e', 'reacher',
                             'hopper', ])
parser.add_argument('-sm', '--search_mode', type=str,
                    default='hpo', choices=['hpo', 'both'])
parser.add_argument('-a', '--alg_name', type=str,
                    default='ppo', choices=['ppo'])
parser.add_argument('-ps', '--pop_size', type=int, default=8,
                    help='the number of configs to suggest for the hyperparameter optimiser.'
                         ' Note that this is not the batch size for the RL training')
parser.add_argument('-mp', '--max_parallel', type=int,
                    default=4, help='max number of parallel brax to spawn')
parser.add_argument('-n', '--exp_name', type=str, default=None)
parser.add_argument('-ni', '--n_init', type=int, default=24)
# Some settings related to PBT-BO
parser.add_argument('-tr', '--t_ready', type=int, default=1_000_000,
                    help='how many steps between explore/exploit')
parser.add_argument('-te', '--t_ready_end', type=int, default=None,
                    help='how many steps between explore/exploit AT THE END.')
parser.add_argument('-td', '--t_distillation', type=int, default=30_000_000,
                    help='how many steps for each distillation')
parser.add_argument('-de', '--distill_every', type=int, default=40_000_000,
                    help='maximum timestep before distillation is triggered. Note that this may happen earlier if the'
                         ' training halts, as determined by the patience parameter')
parser.add_argument('--patience', type=int, default=20, help='number of training steps without improvements before '
                                                             'distillation is triggered.')
parser.add_argument('-mt', '--max_timesteps', type=int, default=150_000_000,
                    help='the maximum timesteps for the master run per phase')
parser.add_argument('-md', '--max_distillation', type=int, default=2)
parser.add_argument('--seed', type=int, default=0,
                    help='the seed of the master run')
parser.add_argument('-qf', '--quantile_fraction', type=float, default=0.125)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-exist', '--existing_policy', type=str,
                    choices=['overwrite', 'resume'], default='resume')
parser.add_argument('--arch_policy', type=str,
                    choices=['random', 'search'], default='search')
args, _ = parser.parse_known_args()

print(args)

# JAX by default pre-allocates 90% of the available VRAM and this will lead to OOM if multiple JAX processes are spawned simultaneously
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.05'

# fix seeds
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# parse the save path and create directories
def main():
    save_path = f'{ROOT_DIR}/data/brax_env_{args.env_name}_{args.alg_name}_maxIter_{args.max_timesteps}_{args.t_ready}_ps_{args.pop_size}_quantileFrac_{args.quantile_fraction}_{args.search_mode}_{args.arch_policy}_bgpbt'
    if args.exp_name is not None:
        save_path = os.path.join(
            save_path, f'{args.exp_name}', f'seed_{args.seed}')
    else:
        save_path = os.path.join(save_path, f'seed_{args.seed}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(
        handlers=[
            logging.StreamHandler()
        ],
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y%m%d %H%M%S',
    )

    env = Brax(env_name=args.env_name,
               alg_name=args.alg_name,
               seed=args.seed,
               do_hpo=True,
               do_nas=args.search_mode == "both",
               log_dir=save_path,
               max_parallel=args.max_parallel,
               )

    if args.search_mode == 'hpo':  # search for hyperparameter only WITHOUT distillation
        hpo = BGPBT(env, log_dir=save_path,
                    max_timesteps=args.max_timesteps,
                    pop_size=args.pop_size,
                    n_init=args.n_init,
                    quantile_fraction=args.quantile_fraction,
                    seed=args.seed,
                    t_ready=args.t_ready,
                    t_ready_end=args.t_ready_end,
                    verbose=args.verbose,
                    existing_policy=args.existing_policy, )
    elif args.search_mode == 'both':
        hpo = BGPBTArch(env, log_dir=save_path,
                        max_timesteps=args.max_timesteps,
                        n_distillation_timesteps=args.t_distillation,
                        pop_size=args.pop_size,
                        n_init=args.n_init,
                        quantile_fraction=args.quantile_fraction,
                        seed=args.seed,
                        t_ready=args.t_ready,
                        t_ready_end=args.t_ready_end,
                        verbose=args.verbose,
                        existing_policy=args.existing_policy,
                        patience=args.patience,
                        distill_every=args.distill_every,
                        max_distillation=args.max_distillation,
                        init_policy='bo' if args.arch_policy == 'search' else 'random', )

    stats = hpo.run()
    stats.to_csv(os.path.join(save_path, f'stats_seed_{args.seed}.csv'))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
