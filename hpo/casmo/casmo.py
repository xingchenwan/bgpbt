from hpo.hpo_base import HyperparameterOptimizer
import ConfigSpace as CS
import torch
import numpy as np
import sys
import math
import gpytorch
from hpo.utils import (
    train_gp,
    copula_standardize,
    interleaved_search,
    get_reward_from_trajectory,
    grad_search
)
from copy import deepcopy
import time
import pandas as pd
import logging
import pickle
import os
from typing import Callable, List

# Some env variables
MAX_CHOLESKY_SIZE = 2000
MIN_CUDA = 1024
# <- here we use cpu only as we sometimes get OOM just from the RL agents.
DEVICE = 'cpu'


class Casmo4RL(HyperparameterOptimizer):

    def __init__(self, env, log_dir,
                 max_iters: int,
                 max_timesteps: int = None,
                 batch_size: int = 1,
                 n_init: int = None,
                 verbose: bool = True,
                 ard=False,
                 use_reward: float = 0.,
                 log_interval: int = 1,
                 time_varying=False,
                 current_timestep: int = 0,
                 acq: str = 'lcb',
                 obj_func: Callable = None,
                 seed: int = None,
                 use_standard_gp: bool = False,
                 ):
        """
        Casmopolitan [Wan2021] with additional support for ordinal variables.

        Args:
            env: an instance of search_spaces.SearchSpace object
            log_dir: path str: the logging directory to save results.
            max_iters: int, maximum number of BO iterations.
            max_timesteps: int, maximum RL timestep.
            batch_size: int, batch size of BO
            n_init: int, number of initializing samples (i.e. random samples)
            ard: whether to use ARD kernel.
            use_reward: bool. When non-zero, we will take the average of the final ``use_reward`` fraction of a 
                reward trajectory as the BO optimization target. Otherwise we only use the final reward.
            log_interval: int. Time interval to save & report the result.
            time_varying: bool whether to use time-varying GP modelling [Bogunovic2016].
            current_timestep: current timestep. Only applicable when time_varying is True
            acq: ['lcb', 'ei']. Choice of the acquisition function.
            obj_func: Callable: the objective function handle.
            seed: random seed.
            use_standard_gp: bool. Whether to use a standard GP. Otherwise we use trust region GP in [Eriksson2019]
                 and [Wan2021].

        References:
        [Bogunovic2016]: Bogunovic, I., Scarlett, J., & Cevher, V. (2016, May). Time-varying Gaussian process bandit optimization. 
            In Artificial Intelligence and Statistics (pp. 314-323). PMLR.
        [Wan2021]: Wan, X., Nguyen, V., Ha, H., Ru, B., Lu, C.,; Osborne, M. A. (2021). 
            Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces. 
            International Conference on Machine Learning. http://arxiv.org/abs/2102.07188
        [Eriksson2019]: Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., & Poloczek, M. (2019). Scalable global optimization via 
            local bayesian optimization. Advances in neural information processing systems, 32.
        """
        super().__init__(env, max_iters, batch_size, 1)
        self.max_timesteps = max_timesteps if max_timesteps is not None else env.default_env_args[
            'num_timesteps']
        # check whether we need to do mixed optimization by inspecting whether there are any continuous dims.
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.verbose = verbose
        self.cur_iters = 0
        self.dim = len(env.config_space.get_hyperparameters())
        self.log_interval = log_interval
        self.n_init = n_init if n_init is not None and n_init > 0 else min(
            10, 2 * self.dim + 1)

        # settings related to the time-varying GP
        self.time_varying = time_varying
        self.current_timestep = current_timestep
        self.use_standard_gp = use_standard_gp

        self.seed = self.env.seed = seed
        self.ard = ard
        self.casmo = _Casmo(env.config_space,
                            n_init=self.n_init,
                            max_evals=self.max_iters,
                            batch_size=None,  # this will be updated later. batch_size=None signifies initialisation
                            verbose=verbose,
                            ard=ard,
                            acq=acq,
                            use_standard_gp=self.use_standard_gp,
                            time_varying=time_varying)
        self.X_init = None
        self.use_reward = use_reward
        # save the RL learning trajectory for each run of the BO
        self.trajectories = []
        self.f = obj_func if obj_func is not None else self._obj_func_handle

    def restart(self):
        self.casmo._restart()
        self.casmo._X = np.zeros((0, self.casmo.dim))
        self.casmo._fX = np.zeros((0, 1))
        self.X_init = np.array([self.env.config_space.sample_configuration(
        ).get_array() for _ in range(self.n_init)])

    def suggest(self, n_suggestions=1, ):
        if self.casmo.batch_size is None:  # Remember the batch size on the first call to suggest
            self.casmo.batch_size = n_suggestions
            self.casmo.n_init = max([self.casmo.n_init, self.batch_size])
            self.restart()

        X_next = np.zeros((n_suggestions, self.dim))

        # Pick from the initial points
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_next[:n_init] = deepcopy(self.X_init[:n_init, :])
            # Remove these pending points
            self.X_init = self.X_init[n_init:, :]

        # Get remaining points from TuRBO
        n_adapt = n_suggestions - n_init
        if n_adapt > 0:
            if len(self.casmo._X) > 0:  # Use random points if we can't fit a GP
                X = deepcopy(self.casmo._X)
                fX = copula_standardize(
                    deepcopy(self.casmo._fX).ravel())  # Use Copula
                X_next[-n_adapt:, :] = self.casmo._create_and_select_candidates(X, fX,
                                                                                length_cont=self.casmo.length,
                                                                                length_cat=self.casmo.length_cat,
                                                                                n_training_steps=100,
                                                                                hypers={}, )[-n_adapt:, :, ]
        suggestions = X_next
        return suggestions

    def suggest_conditional_on_fixed_dims(self, fixed_dims, fixed_vals, n_suggestions=1):
        """Suggest points based on BO surrogate, conditional upon some fixed dims and values"""
        assert len(fixed_vals) == len(fixed_dims)
        X = deepcopy(self.casmo._X)
        fX = copula_standardize(deepcopy(self.casmo._fX).ravel())  # Use Copula
        X_next = self.casmo._create_and_select_candidates(X, fX,
                                                          length_cont=self.casmo.length,
                                                          length_cat=self.casmo.length_cat,
                                                          n_training_steps=100,
                                                          frozen_dims=fixed_dims,
                                                          frozen_vals=fixed_vals,
                                                          batch_size=n_suggestions,
                                                          hypers={}, )
        return X_next

    def observe(self, X, y, t=None):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        t: array-like, shape (n, )
            Corresponding to the timestep vector of t
        """
        assert len(X) == len(y)
        if t is not None:
            assert len(t) == len(y)
        XX = X
        yy = np.array(y)[:, None]
        tt = np.array(t)[:, None] if t is not None else None

        if len(self.casmo._fX) >= self.casmo.n_init:
            self.casmo._adjust_length(yy)

        self.casmo.n_evals += self.batch_size
        self.casmo._X = np.vstack((self.casmo._X, deepcopy(XX)))
        self.casmo._fX = np.vstack(
            (self.casmo._fX, deepcopy(yy.reshape(-1, 1))))
        self.casmo.X = np.vstack((self.casmo.X, deepcopy(XX)))
        self.casmo.fX = np.vstack((self.casmo.fX, deepcopy(yy.reshape(-1, 1))))
        if tt is not None:
            self.casmo._t = np.vstack(
                (self.casmo._t, deepcopy(tt.reshape(-1, 1))))
            self.casmo.t = np.vstack(
                (self.casmo.t, deepcopy(tt.reshape(-1, 1))))

        # Check for a restart
        if self.casmo.length <= self.casmo.length_min:
            self.restart()

    def run(self):
        self.cur_iters = 0
        self.res = pd.DataFrame(np.nan, index=np.arange(self.max_iters + self.batch_size),
                                columns=['Index', 'LastValue', 'BestValue', 'Time'])
        self.X, self.y = [], []
        while self.cur_iters < self.max_iters:
            logging.info(
                f'Current iter = {self.cur_iters + 1} / {self.max_iters}')
            start = time.time()
            suggested_config_arrays = self.suggest(self.batch_size)
            # convert suggestions from np array to a valid configuration.
            suggested_configs = [CS.Configuration(self.env.config_space, vector=array) for array in
                                 suggested_config_arrays]
            rewards = self.f(suggested_configs)
            self.X += suggested_configs
            self.y += rewards
            if isinstance(rewards, float):
                # to give a len to a singleton reward result
                rewards = np.array(rewards).reshape(1)
            self.observe(suggested_config_arrays, rewards)
            end = time.time()
            if len(self.y):
                if self.batch_size == 1:
                    self.res.iloc[self.cur_iters, :] = [self.cur_iters, float(self.y[-1]),
                                                        float(
                                                            np.min(self.y[:self.cur_iters + 1])),
                                                        end - start]
                else:
                    for j in range(self.cur_iters, self.cur_iters + self.batch_size):
                        self.res.iloc[j, :] = [j, float(self.y[j]), float(
                            np.min(self.y[:j + 1])), end - start]
                argmin = np.argmin(self.y[:self.cur_iters + 1])

                logging.info(
                    f'fX={rewards}.'
                    f'fX_best={self.y[argmin]}'
                )
                if self.cur_iters % self.log_interval == 0:
                    if self.log_dir is not None:
                        logging.info(
                            f'Saving intermediate results to {os.path.join(self.log_dir, "stats.pkl")}')
                        self.res.to_csv(os.path.join(
                            self.log_dir, 'stats-pandas.csv'))
                        pickle.dump([self.X, self.y], open(
                            os.path.join(self.log_dir, 'stats.pkl'), 'wb'))
                        pickle.dump(self.trajectories, open(
                            os.path.join(self.log_dir, 'trajectories.pkl'), 'wb'))
            self.cur_iters += self.batch_size

        return self.X, self.y

    def _obj_func_handle(self, config: list, ) -> list:
        """use_synthetic: use the sklearn data generation to generate synthetic functions. """
        trajectories = self.env.train_batch(config, exp_idx_start=self.cur_iters,
                                            nums_timesteps=[
                                                self.max_timesteps] * len(config),
                                            seeds=[self.seed] * len(config),

                                            )
        self.trajectories += trajectories
        reward = [-get_reward_from_trajectory(np.array(t['y']), use_last_fraction=self.use_reward) for t in
                  trajectories]
        return reward

    def get_surrogate(self, current_tr_only=False):
        """Return the surrogate GP fitted on all the training data"""
        if not self.casmo.fX.shape[0]:
            raise ValueError(
                "Casmo does not currently have any observation data!")
        if current_tr_only:
            # the _X and _fX only store the data collected since the last TR restart and got cleared every time after a restart.
            X = deepcopy(self.casmo._X)
            y = deepcopy(self.casmo._fX).flatten()
        else:
            X = deepcopy(self.casmo.X)
            y = deepcopy(self.casmo.fX).flatten()

        ard = self.ard
        if len(X) < self.casmo.min_cuda:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.casmo.device, self.casmo.dtype
        with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(y).to(device=device, dtype=dtype)
            # add some noise to improve numerical stability
            y_torch += torch.randn(y_torch.size()) * 1e-5
            gp = train_gp(
                configspace=self.casmo.cs,
                train_x=X_torch,
                train_y=y_torch,
                use_ard=ard,
                num_steps=100,
                noise_variance=None
            )
        return gp


class _Casmo:
    """A private class adapted from the TurBO code base"""

    def __init__(self, cs: CS.ConfigurationSpace,
                 n_init,
                 max_evals,
                 batch_size: int = None,
                 verbose: bool = True,
                 ard='auto',
                 acq: str = 'ei',
                 time_varying: bool = False,
                 use_standard_gp: bool = False,
                 **kwargs):
        # some env parameters
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        # assert batch_size > 0 and isinstance(batch_size, int)
        if DEVICE == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"
        self.cs = cs
        self.dim = len(cs.get_hyperparameters())
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = ard

        self.acq = acq
        self.kwargs = kwargs
        self.n_init = n_init

        self.time_varying = time_varying

        # hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros(
            (0, self.dim)) if self.use_ard else np.zeros((0, 1))
        self.n_restart = 3  # number of restarts for each acquisition optimization

        # tolerances and counters
        self.n_cand = kwargs['n_cand'] if 'n_cand' in kwargs.keys() else min(
            100 * self.dim, 5000)
        self.use_standard_gp = use_standard_gp
        self.n_evals = 0

        if use_standard_gp:  # this in effect disables any trust region
            logging.info(
                'Initializing a standard GP without trust region or interleaved acquisition search.')
            self.tr_multiplier = 1.
            self.failtol = 100000
            self.succtol = 100000
            self.length_min = self.length_min_cat = -1
            self.length_max = self.length_max_cat = 100000
            self.length_init_cat = self.length_init = 100000

        else:
            self.tr_multiplier = kwargs['multiplier'] if 'multiplier' in kwargs.keys(
            ) else 1.5
            self.failtol = kwargs['failtol'] if 'failtol' in kwargs.keys(
            ) else 10
            self.succtol = kwargs['succtol'] if 'succtol' in kwargs.keys(
            ) else 3

            # Trust region sizes for continuous/int and categorical dimension
            self.length_min = kwargs['length_min'] if 'length_min' in kwargs.keys(
            ) else 0.15
            self.length_max = kwargs['length_max'] if 'length_max' in kwargs.keys(
            ) else 1.
            self.length_init = kwargs['length_init'] if 'length_init' in kwargs.keys(
            ) else .4

            self.length_min_cat = kwargs['length_min_cat'] if 'length_min_cat' in kwargs.keys(
            ) else 0.1
            self.length_max_cat = kwargs['length_max_cat'] if 'length_max_cat' in kwargs.keys(
            ) else 1.
            self.length_init_cat = kwargs['length_init_cat'] if 'length_init_cat' in kwargs.keys(
            ) else 1.

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        # timestep: in case the GP surrogate is time-varying
        self.t = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = MIN_CUDA
        self.dtype = torch.float64
        self.device = torch.device(
            "cuda") if DEVICE == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" %
                  (self.dtype, self.device))
            sys.stdout.flush()

        self._restart()

    def _restart(self):
        self._X = np.zeros((0, self.dim))
        self._fX = np.zeros((0, 1))
        self._t = np.zeros((0, 1))
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init
        self.length_cat = self.length_init_cat

    def _adjust_length(self, fX_next):
        # print(fX_next, self._fX)
        if np.min(fX_next) <= np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += self.batch_size
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += self.batch_size

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min(
                [self.tr_multiplier * self.length, self.length_max])
            self.length_cat = min(
                self.length_cat * self.tr_multiplier, self.length_max_cat)
            self.succcount = 0
            logging.info(f'Expanding TR length to {self.length}')
        elif self.failcount == self.failtol:  # Shrink trust region
            self.failcount = 0
            self.length_cat = max(
                self.length_cat / self.tr_multiplier, self.length_min_cat)
            self.length = max(
                self.length / self.tr_multiplier, self.length_min)
            logging.info(f'Shrinking TR length to {self.length}')

    def _create_and_select_candidates(self, X, fX, length_cat, length_cont,
                                      x_center=None,
                                      n_training_steps=100,
                                      hypers={}, return_acq=False,
                                      time_varying=None,
                                      t=None, batch_size=None,
                                      frozen_vals: list = None,
                                      frozen_dims: List[int] = None):
        d = X.shape[1]
        time_varying = time_varying if time_varying is not None else self.time_varying
        if batch_size is None:
            batch_size = self.batch_size
        if self.use_ard in [True, False]:
            ard = self.use_ard
        else:
            # turn on ARD only when there are many data
            ard = True if fX.shape[0] > 150 else False
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.device, self.dtype
        with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            # add some noise to improve numerical stability
            y_torch += torch.randn(y_torch.size()) * 1e-5

            gp = train_gp(
                configspace=self.cs,
                train_x=X_torch,
                train_y=y_torch,
                use_ard=ard,
                num_steps=n_training_steps,
                hypers=hypers,
                noise_variance=self.kwargs['noise_variance'] if
                'noise_variance' in self.kwargs else None,
                time_varying=time_varying and t is not None,
                train_t=t,
                verbose=self.verbose
            )
            # Save state dict
            hypers = gp.state_dict()

        # we are always optimizing the acquisition function at the latest timestep
        t_center = t.max() if time_varying else None

        def _ei(X, augmented=False):
            """Expected improvement (with option to enable augmented EI).
            This implementation assumes the objective function should be MINIMIZED, and the acquisition function should
                also be MINIMIZED (hence negative sign on both the GP prediction and the acquisition function value)
            """
            from torch.distributions import Normal
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=dtype)
            if X.dim() == 1:
                X = X.reshape(1, -1)
            gauss = Normal(torch.zeros(1), torch.ones(1))
            # flip for minimization problems
            gp.eval()
            if time_varying:
                X = torch.hstack([t_center * torch.ones((X.shape[0], 1)), X])
            preds = gp(X)
            with gpytorch.settings.fast_pred_var():
                mean, std = -preds.mean, preds.stddev
            mu_star = -fX.min()

            u = (mean - mu_star) / std
            ucdf = gauss.cdf(u)
            updf = torch.exp(gauss.log_prob(u))
            ei = std * updf + (mean - mu_star) * ucdf
            if augmented:
                sigma_n = gp.likelihood.noise
                ei *= (1. - torch.sqrt(torch.clone(sigma_n)) /
                       torch.sqrt(sigma_n + std ** 2))
            return -ei

        def _lcb(X, beta=3.):
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=dtype)
            if X.dim() == 1:
                X = X.reshape(1, -1)
            if time_varying:
                X = torch.hstack([t_center * torch.ones((X.shape[0], 1)), X])
            gp.eval()
            gp.likelihood.eval()
            preds = gp.likelihood(gp(X))
            with gpytorch.settings.fast_pred_var():
                mean, std = preds.mean, preds.stddev
                lcb = mean - beta * std
            return lcb

        if batch_size == 1:
            # Sequential setting
            if self.use_standard_gp:
                X_next, acq_next = grad_search(self.cs, x_center[0] if x_center is not None else None, eval(f'_{self.acq}'),
                                               n_restart=self.n_restart, batch_size=batch_size,
                                               verbose=self.verbose,
                                               dtype=dtype)
            else:
                X_next, acq_next = interleaved_search(self.cs,
                                                      d,
                                                      x_center[0] if x_center is not None else None,
                                                      eval(f'_{self.acq}'),
                                                      max_dist_cat=length_cat,
                                                      max_dist_cont=length_cont,
                                                      cont_int_lengthscales=gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel(),
                                                      n_restart=self.n_restart, batch_size=batch_size,
                                                      verbose=self.verbose,
                                                      frozen_dims=frozen_dims,
                                                      frozen_vals=frozen_vals,
                                                      dtype=dtype)

        else:
            # batch setting: for these, we use the fantasised points {x, y}
            X_next = torch.tensor([], dtype=dtype, device=device)
            acq_next = np.array([])
            for p in range(batch_size):
                x_center_ = deepcopy(
                    x_center[0]) if x_center is not None else None
                if self.use_standard_gp:
                    x_next, acq = grad_search(self.cs, x_center_, eval(f'_{self.acq}'),
                                              n_restart=self.n_restart, batch_size=1,
                                              dtype=dtype)
                else:
                    x_next, acq = interleaved_search(self.cs,
                                                     d,
                                                     x_center_,
                                                     eval(f'_{self.acq}'),
                                                     max_dist_cat=length_cat,
                                                     max_dist_cont=length_cont,
                                                     cont_int_lengthscales=gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel(),
                                                     frozen_dims=frozen_dims,
                                                     frozen_vals=frozen_vals,
                                                     n_restart=self.n_restart, batch_size=1, dtype=dtype)

                x_next_torch = torch.tensor(x_next, dtype=dtype, device=device)
                if time_varying:
                    # strip the time dimension
                    x_next_torch = x_next_torch[:, 1:]

                y_next = gp(x_next_torch).mean.detach()
                with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
                    X_torch = torch.cat((X_torch, x_next_torch), dim=0)
                    y_torch = torch.cat((y_torch, y_next), dim=0)
                    gp = train_gp(
                        configspace=self.cs,
                        train_x=X_torch, train_y=y_torch, use_ard=ard, num_steps=n_training_steps,
                        hypers=hypers,
                        noise_variance=self.kwargs['noise_variance'] if
                        'noise_variance' in self.kwargs else None,
                        time_varying=self.time_varying,
                        train_t=t,
                    )
                X_next = torch.cat((X_next, x_next_torch), dim=0)
                acq_next = np.hstack((acq_next, acq))

        del X_torch, y_torch, gp
        X_next = np.array(X_next)
        if return_acq:
            return X_next, acq_next
        return X_next
