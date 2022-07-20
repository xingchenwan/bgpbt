from hpo.casmo.casmo import Casmo4RL, MAX_CHOLESKY_SIZE, MIN_CUDA
from hpo.utils import normalize, copula_standardize, train_gp, get_reward_from_trajectory, is_large
import copy
import random
import shutil
import pandas as pd
import numpy as np
import ConfigSpace as CS
import gpytorch
import torch
import logging
from collections import OrderedDict
import os


class BGPBT(Casmo4RL):

    def __init__(self, env, log_dir,
                 max_timesteps: int = None,
                 pop_size: int = 4,
                 n_init: int = None,
                 verbose: bool = False,
                 ard=False,
                 t_ready: int = None,
                 quantile_fraction: float = .25,
                 seed: int = None,
                 use_reward_fraction: float = 0.,
                 existing_policy: str = 'overwrite',
                 backtrack: bool = True,
                 schedule_t_ready: bool = False,
                 t_ready_end: int = None,
                 guided_restart: bool = False,
                 ):
        """
        BGPBT with Casmopolitan [Wan2021] surrogate without distillation & NAS.

        Note: this optimizer expects a minimization problem. Remember to negate the objective if you have a maximisation
            problem! (e.g. maximising a RL reward)

        Args:
            env: an instance of search_spaces.SearchSpace object
            log_dir: path str: the logging directory to save results.
            pop_size: int: population size of PBT.
            n_init: int: number of randomly initialising points. While selecting points within the n_init budget, the GP surrogate
                will not be called.
            ard: whether enable automatic relevance selection.
            t_ready: int, 0 < t_ready_start < max_timesteps: the interval expressed in terms of timesteps to tune the hyperparameters
                (and optionally the architectures)
            quantile_fraction: float, 0 < quantile_fraction < 0.5. The fraction of agents that will be replaced during each
                PBT exploitation step
            seed: int: random seed -- will be used to set both the HPO and the environment.
            existing_policy: str. options=["resume", "overwrite"] -- what to do when we have a log_dir with existing files (mostly due to a previous run).
            t_ready_end: the ending t_ready_start value. In case it is different from the t_ready_start, a linear
                schedule will be used across them.
            guided_restart: bool. Whether use the auxiliary GP restart style presented in [Wan2021]. Otherwise will sipmly use
                standard random restart.

        References:
            [Wan2021]: Wan, X., Nguyen, V., Ha, H., Ru, B., Lu, C.,; Osborne, M. A. (2021). 
            Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces. 
            International Conference on Machine Learning. http://arxiv.org/abs/2102.07188
        """
        super().__init__(env, log_dir=log_dir, max_iters=1, max_timesteps=max_timesteps,
                         verbose=verbose, ard=ard, n_init=n_init, use_reward=use_reward_fraction)
        assert 0 < quantile_fraction <= .5
        assert int(quantile_fraction *
                   pop_size) >= 1, 'quantile_fraction * pop size must be >= 1!'
        assert 0 < t_ready < max_timesteps
        self.pop_size = pop_size
        self.min_pop_size = max(int(round(0.5 * self.pop_size)), 1)

        self.t_ready_start = t_ready
        self.t_ready_end = t_ready_end
        self.schedule_t_ready = schedule_t_ready

        self.quantile_fraction = quantile_fraction
        self.seed = self.env.seed = seed
        self.backtrack = backtrack
        self.n_distills = 0
        self.budget_type = 'timestep_total'
        self.df = pd.DataFrame(
            columns=['Agent', 't', self.budget_type, 'R', 'R_test', 'conf', 'path', 'conf_', 'config_source',
                     'excluded', 'n_distills', 'policy_net', 'value_net'])

        # default params for action and value nets
        self.policy_net = [32, 32, 32, 32]
        self.value_net = [256, 256, 256, 256, 256]
        checkpoint_dir = f'{self.log_dir}/pb2_checkpoints'
        self.best_checkpoint_dir = f'{self.log_dir}/pb2_checkpoints/best_checkpoint.pt'
        self.best_cost = np.inf
        self.patience = 15
        self.n_fail = 0

        # whether the run is resumed from previous
        self.resumed = False

        # initialise an empty OrderedDict first (to be updated either by loading from disk or search_init)
        self.pop = OrderedDict({i: {'done': False,
                                    'config': self.env.config_space.sample_configuration(),
                                    'path': os.path.join(checkpoint_dir,
                                                         f'{self.env.env_name}_seed{self.env.seed}_Agent{i}.pt'),
                                    'config_source': 'random',
                                    'excluded': False,
                                    'distill': False}
                                for i in range(self.pop_size)})

        if not os.path.exists(checkpoint_dir):
            logging.info(f'Creating directory={checkpoint_dir}')
            os.makedirs(checkpoint_dir)
        elif existing_policy == 'overwrite':
            logging.info(
                f'Checkpoint directory {checkpoint_dir} already exists and I am instructed overwrite. Deleting the old folder.')
            shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir)
        elif existing_policy == 'resume':
            # loading from existing
            logging.info(f'Resuming from {checkpoint_dir}')
            df_file = f'{self.log_dir}/stats_seed_{self.seed}_intermediate.csv'
            assert os.path.exists(df_file), f'Required file {df_file} missing!'
            self.df = pd.read_csv(df_file)
            self.df = self.df[
                ['Agent', 't', self.budget_type, 'R', 'R_test', 'conf', 'path', 'conf_', 'config_source', 'excluded',
                 'n_distills', 'policy_net', 'value_net']]
            self.policy_net = eval(self.df.policy_net.iloc[-1])
            self.value_net = eval(self.df.value_net.iloc[-1])
            self.df.conf = self.df.conf.apply(lambda x: eval(x))
            self.n_distills = self.df.n_distills.max()
            for agent, params in self.pop.items():
                max_t = self.df[self.df['Agent'] ==
                                agent][self.budget_type].max()
                agent_info = self.df[(self.df['Agent'] == agent) & (
                    self.df[self.budget_type] == max_t)].iloc[-1]
                self.pop[agent]['done'] = agent_info[self.budget_type] >= self.max_timesteps
                self.pop[agent]['config'] = CS.Configuration(
                    self.env.config_space, vector=np.array(agent_info['conf']))
                self.pop[agent]['path'] = agent_info['path']
                self.pop[agent]['config_source'] = agent_info['config_source']
                self.pop[agent]['excluded'] = agent_info['excluded']
            # setup the best agent
            if self.backtrack and self.env.env_name not in ['dummy', 'synthetic']:
                self.best_cost = self.df[self.df.n_distills ==
                                         self.n_distills]['R'].min()
                overall_best_agent = \
                    self.df[(self.df['R'] == self.best_cost) & (
                        self.df['n_distills'] == self.n_distills)].iloc[-1]
                shutil.copy(
                    overall_best_agent['path'], self.best_checkpoint_dir)
            self.resumed = True
        else:
            raise ValueError(f"Unknown existing_policy: {existing_policy}")

        self.running = {}
        self.guided_restart = guided_restart

    def search_init(self):
        """Search for a good initialization by doing end-to-end (i.e. non-population based) BO for a short timeframe"""
        self.init_idx = 0
        if self.guided_restart and self.n_distills > 0:
            init_configs = self._generate_initializing_points_ucb(
                init_size=max(self.n_init, self.pop_size))
        else:
            init_configs = [self.env.config_space.sample_configuration(
            ) for _ in range(max(self.n_init, self.pop_size))]
        init_store_paths = [os.path.join(f'{self.log_dir}/pb2_checkpoints',
                                         f'{self.env.env_name}_seed{self.env.seed}_InitConfig{i}_Stage{self.n_distills}.pt')
                            for i in range(len(init_configs))]
        self.init_results_tmp = self.env.train_batch(configs=init_configs, seeds=[self.seed] * len(init_configs),
                                                     nums_timesteps=[
                                                         self.t_ready_start] * len(init_configs),
                                                     checkpoint_paths=init_store_paths,
                                                     policy_hidden_layer_sizes=self.policy_net,
                                                     v_hidden_layer_sizes=self.value_net,
                                                     )
        costs = [-get_reward_from_trajectory(r['y'], self.use_reward, 0.)
                 for r in self.init_results_tmp]

        if self.n_init <= self.pop_size:
            top_config_ids = np.arange(len(init_configs)).tolist()
        # we sample more configs (BO or random sampling), and only start PBT using the best of those.
        else:
            # using the ``pop_size'' best as the initialising population
            top_config_ids = np.argpartition(np.array(costs), self.pop_size)[
                :self.pop_size].tolist()
        for i, config_id in enumerate(top_config_ids):
            self.pop[i] = {
                'done': False,
                'config': init_configs[config_id],
                'path': os.path.join(f'{self.log_dir}/pb2_checkpoints',
                                     f'{self.env.env_name}_seed{self.env.seed}_Agent{i}.pt'),
                'config_source': 'random',
                'excluded': False,
            }
            shutil.copy(os.path.join(f'{self.log_dir}/pb2_checkpoints',
                                     f'{self.env.env_name}_seed{self.env.seed}_InitConfig{config_id}_Stage{self.n_distills}.pt'),
                        os.path.join(f'{self.log_dir}/pb2_checkpoints',
                                     f'{self.env.env_name}_seed{self.env.seed}_Agent{i}.pt'))
        for path in init_store_paths:
            os.remove(path)  # delete the unused checkpoints
        # update the dataframe with these data (including those bad-performing points not selected in the initial population)
        t = 1 if self.n_distills == 0 else self.df.t.max() + 1
        for i in range(len(init_configs)):
            config = init_configs[i]
            conf_array = config.get_array()
            final_cost = costs[i]
            rl_reward = get_reward_from_trajectory(
                self.init_results_tmp[i]['y'], 1, 0.)
            scalar_steps = self.init_results_tmp[i]['x'][-1]
            d = pd.DataFrame(columns=self.df.columns)
            agent_number = - \
                1 if i not in top_config_ids else top_config_ids.index(i)
            path = self.pop[agent_number]['path'] if agent_number >= 0 else np.nan
            d.loc[0] = [agent_number, t, scalar_steps, final_cost, rl_reward, conf_array.tolist(), path, config, 'random', False,
                        self.n_distills, self.policy_net, self.value_net]
            self.df = pd.concat([self.df, d]).reset_index(drop=True)
            logging.info("\nAgent: {}, Timesteps: {}, Cost: {}\n".format(
                agent_number, scalar_steps, final_cost, ))
        del self.init_results_tmp, self.init_idx
        return self.pop

    def run(self, ):
        # conf is the internal representation of the array: with continuous/integer hyperparameteres scaled between 0 and 1.
        # conf_ is a ConfigSpace representation that is human-readable.
        all_done = False
        if not self.resumed:
            logging.info('Searching for initialising configurations!')
            self.search_init()
        # specify the checkpoint path of all agents
        while not all_done:
            if self.df.t.empty:
                t = 1
            else:
                t = self.df.t.max() + 1

            non_excluded_pop = OrderedDict(
                {k: v for k, v in self.pop.items() if not v['excluded']})

            # avoid OOM by detecting whether there are big unroll length agents -- if >50% of the agents have unroll
            # length > 20, halve the parallel count
            n_large_models = sum([is_large(c['config'])
                                 for c in self.pop.values()])
            max_parallel = self.env.max_parallel // 2 if n_large_models / len(
                non_excluded_pop) > 0.5 else self.env.max_parallel
            logging.info(f'Max parallel for this iteration={max_parallel}')

            if self.t_ready_end == self.t_ready_start or self.t_ready_end is None:
                t_ready = self.t_ready_start
            else:
                t_ready = int(self.t_ready_start + (self.t_ready_end - self.t_ready_start) /
                              self.max_timesteps * self.df[self.budget_type].max())

            results_values = self.env.train_batch(configs=[c['config'] for c in non_excluded_pop.values()],
                                                  seeds=[self.seed] *
                                                  len(non_excluded_pop),
                                                  nums_timesteps=[
                                                      t_ready] * len(non_excluded_pop),
                                                  checkpoint_paths=[
                                                      c['path'] for c in non_excluded_pop.values()],
                                                  max_parallel=max_parallel)
            results_keys = list(non_excluded_pop.keys())
            results = dict(zip(results_keys, results_values))
            for agent in self.pop.keys():
                if self.pop[agent]['done']:
                    logging.info(f'Skipping completed agent {agent}.')
                    continue
                final_cost = - \
                    get_reward_from_trajectory(
                        results[agent]['y'], self.use_reward, 0.)
                rl_reward = get_reward_from_trajectory(
                    results[agent]['y'], 1, 0.)
                final_timestep = results[agent]['x'][-1]

                if self.df[self.df['Agent'] == agent].empty:
                    scalar_steps = final_timestep
                else:
                    scalar_steps = final_timestep + \
                        self.df[self.df['Agent'] ==
                                agent][self.budget_type].max()
                logging.info("\nAgent: {}, Timesteps: {}, Cost: {}\n".format(
                    agent, scalar_steps, final_cost))

                conf_array = self.pop[agent]['config'].get_array().tolist()
                conf = self.pop[agent]['config']
                config_source = self.pop[agent]['config_source']
                d = pd.DataFrame(columns=self.df.columns)
                d.loc[0] = [agent, t, scalar_steps, final_cost, rl_reward, conf_array,
                            self.pop[agent]['path'], conf, config_source, self.pop[agent]['excluded'], self.n_distills,
                            self.policy_net, self.value_net]
                self.df = pd.concat([self.df, d]).reset_index(drop=True)

                if self.df[self.df['Agent'] == agent][self.budget_type].max() >= self.max_timesteps:
                    self.pop[agent]['done'] = True

            # update the trust region based on the results of the agents from previous runs, before exploitation
            self.adjust_tr_length(restart=True)

            if self.backtrack and self.env.env_name not in ['dummy', 'synthetic']:
                best_cost = self.df['R'].min()
                if best_cost < self.best_cost:
                    self.best_cost = best_cost
                    overall_best_agent = self.df[self.df['R']
                                                 == best_cost].iloc[-1]
                    shutil.copy(
                        overall_best_agent['path'], self.best_checkpoint_dir)

            non_excluded_pop = OrderedDict(
                {k: v for k, v in self.pop.items() if not v['excluded']})
            # exploitation -- copy the weights and etc.
            for agent in non_excluded_pop.keys():
                old_conf = self.pop[agent]['config'].get_array().tolist()
                self.pop[agent], copied = self.exploit(agent, )
                # here we need to include a way to account for changes in the data.
                new_conf = self.pop[agent]['config'].get_array().tolist()
                if not np.isclose(0, np.nansum(np.array(old_conf) - np.array(new_conf))):
                    logging.info("changing conf for agent: {}".format(agent))
                    new_row = self.df[(self.df['Agent'] == copied) & (
                        self.df['t'] == self.df.t.max())]
                    new_row['Agent'] = agent
                    # new_row['path'] = self.pop[agent]['path']
                    logging.info(f"new row conf old: {new_row['conf']}")
                    logging.info(f"new row conf new: {[new_conf]}")
                    new_row['conf'] = [new_conf]
                    new_row['conf_'] = [CS.Configuration(
                        self.env.config_space, vector=new_conf)]
                    new_row['excluded'] = self.pop[agent]['excluded']
                    self.df = pd.concat(
                        [self.df, new_row]).reset_index(drop=True)
                    logging.info(f"new config: {new_conf}")

            all_done = np.array([self.pop[agent]['done']
                                for agent in self.pop.keys()]).all()
            # save intermediate results
            self.df.to_csv(os.path.join(
                self.log_dir, f'stats_seed_{self.seed}_intermediate.csv'))

            t_max = self.df[self.budget_type].max()
            best_loss = self.df[self.df['n_distills']
                                == self.n_distills].R.min()
            if self.df[
                    (self.df[self.budget_type] == t_max) & (self.df['n_distills'] == self.n_distills)].R.min() == best_loss:
                self.n_fail = 0
            else:
                self.n_fail += 1
            # restart when the casmo trust region is below threshold
            if self.n_fail >= self.patience:
                self.n_fail = 0
                logging.info('n_fail reached patience. Restarting GP')
                self._restart()
            logging.info(f'n_fail: {self.n_fail}')

        return self.df

    def exploit(self, agent):
        # when just doing HPO, we can simply use the original PBT type exploitation
        if self.df[self.df['Agent'] == agent].t.empty:
            return self.pop[agent]
        else:
            n = max(int(self.pop_size * self.quantile_fraction), 1)
            max_t = self.df.t.max()  # last iteration entry
            last_entries = self.df[(self.df['t'] == max_t) & (
                self.df['excluded'] == 0)]  # index entire population based on last set of runs
            # only want the original entries
            last_entries = last_entries.iloc[:self.pop_size]
            ranked_last_entries = last_entries.sort_values(
                by=['R'], ignore_index=True, ascending=False)  # rank last entries
            position = list(ranked_last_entries.Agent.values).index(
                agent) + 1  # not indexed to zero
            if position <= n and len(ranked_last_entries) > self.min_pop_size:
                # the agent is a "bad" config and will be replaced in weights by a new config
                best_agents = list(
                    ranked_last_entries.iloc[-n:]['Agent'].values)
                best_agent = random.sample(best_agents, 1)[0]

                new_config, new_config_source = self.explore(
                    agent, best_agent, )
                self.pop[agent]['config'] = new_config
                self.pop[agent]['config_source'] = new_config_source
                logging.info(
                    "\n replaced agent {} with agent {}".format(agent, best_agent))
                logging.info(self.pop[agent]['config'])

                if self.env.env_name not in ['synthetic', 'dummy']:
                    best_path = self.best_checkpoint_dir if self.backtrack else self.pop[
                        best_agent]['path']
                    current_path = self.pop[agent]['path']
                    shutil.copy(best_path, current_path)
                    self.pop[agent]['excluded'] = False

            else:
                # not exploiting, not exploring... move on :)
                logging.info(f'Continuing training for agent {agent}.')
                best_agent = copy.copy(agent)
                self.pop[agent]['config_source'] = 'previous'

            return self.pop[agent], best_agent

    def explore(self, agent, best_agent):
        """Fit a Casmo model to the existing data and run BayesOpt to suggest new configurations."""
        dfnewpoint, data, agent_t = self.format_df(agent, best_agent)
        # dfnewpoint contains the information of the previous best agent. At the explore step, we replace it with an
        #   alternative hparam config.
        if (not dfnewpoint.empty) and data.shape[0] >= self.pop_size:
            y = np.array(data.y.values)
            t = np.array(data.t.values)
            # hyperparameter dimensions -- active dimensions that will be tuned.
            hparams = data[['x{}'.format(i)
                            for i in range(len(self.df.conf[0]))]]
            # contextual dimension
            # config array of the running parameters
            current = np.array(
                [x for x in self.running[str(agent_t)].values()])
            # concatenate the array with the contextual dimensions
            t_current = np.tile(
                np.array(dfnewpoint.t.values), current.shape[0])

            # it is important that the contextual information is appended to the end of the vector.
            t_r = data[
                ["R_before"]]  # fixed dimensions -- serve as contextual information for BO but may not be modified.
            X = pd.concat([hparams, t_r], axis=1).values
            t_r_current = np.tile(
                dfnewpoint[["R_before"]].values, (current.shape[0], 1))
            current = np.hstack([current, t_r_current]).astype(float)
            # get the hp of the best agent selected from -- this will be trust region centre
            X_best = dfnewpoint[
                ['x{}'.format(i) for i in range(len(self.df.conf[0]))] + ["R_before"]].values
            new_config_array = self.select_config(
                X, y, t, current, t_current, x_center=X_best)
            new_config = CS.Configuration(
                self.env.config_space, vector=new_config_array)
            config_source = 'bo'
        else:
            logging.info('Using random exploration.')
            new_config = self.env.config_space.sample_configuration()
            config_source = 'random'
        to_use = new_config.get_array().tolist()

        try:
            self.running[str(agent_t)].update({str(agent): to_use})
        except KeyError:
            self.running.update({str(agent_t): {str(agent): to_use}})

        return new_config, config_source

    def select_config(self, X, y, t, X_current=None, t_current=None, x_center=None, frozen_dims=None, frozen_vals=None):
        """Main BO Loop (corresponding to the self.suggest function in Casmo.py.
        current: denotes the running configurations of the agent, which are added to the GP with a dummy output to
            ensure the proposed configurations do not lie close to these pending configs.
        X, y: evaluated X and y data
        frozen_dims and frozen_vals: the frozen dimension indices and their values that will not be optimized.
        """
        # 1. normalize the fixed dimensions (note that the variable dimensions are already scaled to [0,1]^d using config_space
        # sometimes we get object array and cast them to float
        X = np.array(X).astype(float)
        y = np.array(y).astype(float)
        if t is not None:
            t = np.array(t).astype(float)
        if X_current is not None:
            X_current = np.array(X_current).astype(float)
        if t_current is not None:
            t_current = np.array(t_current).astype(float)
        # the dimensions attached to the end of the vector are fixed dims
        num_fixed = X.shape[1] - len(self.env.config_space)
        if num_fixed > 0:
            oldpoints = X[:, -num_fixed:]  # fixed dimensions
            # appropriate rounding
            if X_current is not None:
                newpoint = X_current[:, -num_fixed:]
                fixed_points = np.concatenate((oldpoints, newpoint), axis=0)
            else:
                fixed_points = oldpoints
            lims = np.concatenate((np.max(fixed_points, axis=0), np.min(
                fixed_points, axis=0))).reshape(2, oldpoints.shape[1])

            lims[0] -= 1e-8
            lims[1] += 1e-8

            X[:, -num_fixed:] = normalize(X[:, -num_fixed:], lims)
        hypers = {}
        use_time_varying_gp = np.unique(t).shape[0] > 1
        if x_center is not None and num_fixed > 0:
            x_center[:, -
                     num_fixed:] = normalize(x_center[:, -num_fixed:], lims)
        if X_current is not None:
            # 2. Train a GP conditioned on the *real* data which would give us the fantasised y output for the pending fixed_points
            if num_fixed > 0:
                X_current[:, -
                          num_fixed:] = normalize(X_current[:, -num_fixed:], lims)
            y = copula_standardize(copy.deepcopy(y).ravel())
            if len(X) < MIN_CUDA:
                device, dtype = torch.device("cpu"), torch.float32
            else:
                device, dtype = self.casmo.device, self.casmo.dtype

            with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
                X_torch = torch.tensor(X).to(device=device, dtype=dtype)
                # here we replace the nan values with zero, but record the nan locations via the X_torch_nan_mask
                y_torch = torch.tensor(y).to(device=device, dtype=dtype)
                # add some noise to improve numerical stability
                y_torch += torch.randn(y_torch.size()) * 1e-5
                t_torch = torch.tensor(t).to(device=device, dtype=dtype)

                gp = train_gp(
                    configspace=self.casmo.cs,
                    train_x=X_torch,
                    train_y=y_torch,
                    use_ard=False,
                    num_steps=200,
                    time_varying=True if use_time_varying_gp else False,
                    train_t=t_torch,
                    verbose=self.verbose
                )
                hypers = gp.state_dict()
            # 3. Get the posterior prediction at the fantasised points
            gp.eval()
            if use_time_varying_gp:
                t_x_current = torch.hstack(
                    (torch.tensor(t_current, dtype=dtype).reshape(-1, 1), torch.tensor(X_current, dtype=dtype)))
            else:
                t_x_current = torch.tensor(X_current, dtype=dtype)
            pred_ = gp(t_x_current).mean
            y_fantasised = pred_.detach().numpy()
            y = np.concatenate((y, y_fantasised))
            X = np.concatenate((X, X_current), axis=0)
            t = np.concatenate((t, t_current))
            del X_torch, y_torch, t_torch, gp

        # scale the fixed dimensions to [0, 1]^d
        y = copula_standardize(copy.deepcopy(y).ravel())
        # simply call the _create_and_select_candidates subroutine to return
        next_config = self.casmo._create_and_select_candidates(X, y, length_cat=self.casmo.length_cat,
                                                               length_cont=self.casmo.length,
                                                               hypers=hypers, batch_size=1,
                                                               t=t if use_time_varying_gp else None,
                                                               time_varying=use_time_varying_gp,
                                                               x_center=x_center,
                                                               frozen_dims=frozen_dims,
                                                               frozen_vals=frozen_vals,
                                                               n_training_steps=1).flatten()
        # truncate the array to only keep the hyperparameter dimenionss
        if num_fixed > 0:
            next_config = next_config[:-num_fixed]
        return next_config

    def format_df(self, agent, best_agent_selected=None, n_distills=None):
        """
        Helper func for PB2 methods.
        Input: args, the agent index, and total df
        Output: dfnewpoint: New fixed params, data: formatted data
        """
        # Get current
        if n_distills is None:
            n_distills = self.n_distills
        n = max(int(self.pop_size * self.quantile_fraction), 1)
        agent_t = self.df[self.df['Agent'] ==
                          agent].t.max()  # last iteration entry
        # index entire population based on last set of runs
        last_entries = self.df[self.df['t'] == agent_t]
        ranked_last_entries = last_entries.sort_values(by=['R'], ignore_index=True,
                                                       ascending=False)  # rank last entries
        # best_agents = list(ranked_last_entries.iloc[-n:]['Agent'].values)

        # log the current config of the other running agents whose weights will not be replaced in the current round,
        # and do not sample near them.
        not_exploring = list(ranked_last_entries.iloc[:-n]['Agent'].values)
        for a in not_exploring:
            try:
                self.running[str(agent_t)].update(
                    {str(a): self.df[(self.df['Agent'] == a) & (self.df['t'] == agent_t)]['conf'].values[0]})
            except KeyError:
                self.running.update(
                    {str(agent_t): {
                        str(a): self.df[(self.df['Agent'] == a) & (self.df['t'] == agent_t)]['conf'].values[0]}})

        data = self.df[['Agent', 't', self.budget_type, 'R', 'n_distills']]
        # whether to constrain the data to the current n_distills
        data = data[data['n_distills'] == n_distills]
        if data.shape[0] == 0:   # empty
            return data, data, agent_t,
        data = data.dropna()
        # the configuration array
        data[['x{}'.format(i) for i in range(len(self.df.conf[0]))]] = pd.DataFrame(self.df.conf.tolist(),
                                                                                    index=self.df.index)

        data["y"] = data.groupby(["Agent"] + ['x{}'.format(i)
                                 for i in range(len(self.df.conf[0]))])["R"].diff()
        data["t_change"] = data.groupby(["Agent"] + ['x{}'.format(i) for i in range(len(self.df.conf[0]))])[
            self.budget_type].diff()

        data = data[data["t_change"] > 0].reset_index(drop=True)
        data["R_before"] = data.R - data.y

        data["y"] = data.y / data.t_change
        # the first row is now NaN
        data = data[~data.y.isna()].reset_index(drop=True)
        data = data.sort_values(by=self.budget_type).reset_index(drop=True)
        # when data is too big gpytorch throws random errors.
        data = data.iloc[-100:, :].reset_index(drop=True)
        # the fixed dimension for the next round of selection
        if best_agent_selected is not None:
            dfnewpoint = data[(data["Agent"] == best_agent_selected) & (
                data['t'] == agent_t)]
            return dfnewpoint, data, agent_t
        return data, agent_t

    def adjust_tr_length(self, restart=False):
        """Adjust trust region size -- the criterion is that whether any config sampled by BO outperforms the other config
        sampled otherwise (e.g. randomly, or carried from previous timesteps). If true, then it will be a success or
        failure otherwise."""
        df = self.df.copy()
        t_max = df.t.max()  # last iteration entry
        agents = df[df.t == t_max]
        # get the negative reward
        best_reward = np.min(agents.R.values)
        # get the agents selected by Bayesian optimization
        bo_agents = agents[agents.config_source == 'bo']
        if bo_agents.shape[0] == 0:
            return
        # if the best reward is caused by a config suggested by BayesOpt
        if np.min(bo_agents.R.values) == best_reward:
            self.casmo.succcount += 1
            self.casmo.failcount = 0
        else:
            self.casmo.failcount += 1
            self.casmo.succcount = 0
        if self.casmo.succcount == self.casmo.succtol:  # Expand trust region
            self.casmo.length = min(
                [self.casmo.tr_multiplier * self.casmo.length, self.casmo.length_max])
            self.casmo.length_cat = min(
                self.casmo.length_cat * self.casmo.tr_multiplier, self.casmo.length_max_cat)
            self.casmo.succcount = 0
            logging.info(f'Expanding TR length to {self.casmo.length}')
        elif self.casmo.failcount == self.casmo.failtol:  # Shrink trust region
            self.casmo.failcount = 0
            self.casmo.length_cat = max(
                self.casmo.length_cat / self.casmo.tr_multiplier, self.casmo.length_min_cat)
            self.casmo.length = max(
                self.casmo.length / self.casmo.tr_multiplier, self.casmo.length_min)
            logging.info(f'Shrinking TR length to {self.casmo.length}')
        if restart and (self.casmo.length <= self.casmo.length_min or self.casmo.length_max_cat <= self.casmo.length_min_cat):
            self._restart()

    def _restart(self):
        logging.info('Restarting!')
        self.n_distills += 1  # this will cause the GP to reset in the next iteration
        self.casmo.length = self.casmo.length_init
        self.casmo.length_cat = self.casmo.length_init_cat
        self.casmo.failcount = self.casmo.succcount = 0

    def _generate_initializing_points_ucb(self, n_init):
        # for each of the previous restart (for restart > 0), based on the current GP, find the best points based on
        #    the auxiliary GP, ranked by their UCB score -- this is required for the Theoretical guarantee but is
        #    only applicable in the case without distillation and etc.
        if n_init is None:
            n_init = self.n_init
        # fit a GP based on the results from the previous restart
        # this will return the data in the latest n_distills
        dfnewpoint, data, _ = self.format_df(0,)
        y = np.array(data.y.values).astype(float)
        t = np.array(data.t.values).astype(float)
        X = data[['x{}'.format(i) for i in range(len(self.df.conf[0]))]]
        y = copula_standardize(copy.deepcopy(y).ravel())
        if len(X) < MIN_CUDA:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.casmo.device, self.casmo.dtype

        with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(y).to(device=device, dtype=dtype)
            # add some noise to improve numerical stability
            y_torch += torch.randn(y_torch.size()) * 1e-5
            t_torch = torch.tensor(t).to(device=device, dtype=dtype)

            gp = train_gp(
                configspace=self.casmo.cs,
                train_x=X_torch,
                train_y=y_torch,
                use_ard=False,
                num_steps=200,
                time_varying=True,
                train_t=t_torch,
                verbose=self.verbose
            )
        gp.eval()
        # the training points to add for the auxiliary GP
        aux_train_input, aux_train_target = [], []
        for restart in range(self.n_distills):
            dfnewpoint, data, _ = self.format_df(0, n_distills=restart)
            X = data[['x{}'.format(i) for i in range(len(self.df.conf[0]))]]
            t_current = np.max(t) * np.ones(X.shape[0])
            t_x_current = torch.hstack(
                (torch.tensor(t_current, dtype=dtype).reshape(-1, 1), torch.tensor(X, dtype=dtype)))
            pred_ = gp(t_x_current).mean
            # select the x with the best
            best_idx = np.argmin(pred_.detach().numpy())
            aux_train_input.append(X[best_idx, :])
            aux_train_target.append(pred_.detach().numpy()[best_idx, :])
        # now fit the auxiliary GP
        aux_gp = train_gp(
            configspace=self.casmo.cs,
            train_x=torch.tensor(aux_train_input).to(
                device=device, dtype=dtype),
            train_y=torch.tensor(aux_train_target).to(
                device=device, dtype=dtype),
            use_ard=False,
            num_steps=200,
            time_varying=True,
            train_t=t_torch,
            verbose=self.verbose
        )
        aux_gp.eval()

        # now generate a bunch of random configs
        random_configs = [self.env.config_space.sample_configuration(
        ).get_array() for _ in range(10 * n_init)]
        random_config_arrays = [c.get_array() for c in random_configs]
        t_current = np.max(t) * np.ones(len(random_config_arrays))

        # selection by the UCB score using the predicted mean + var of the auxiliary GP.
        random_config_array_t = torch.hstack(
            (torch.tensor(t_current, dtype=dtype).reshape(-1, 1), torch.tensor(random_config_arrays, dtype=dtype)))
        pred = aux_gp(random_config_array_t)
        pred_mean, pred_std = pred.mean.detach().numpy(), pred.stddev.detach().numpy()
        ucb = pred_mean - 1.96 * pred_std
        top_config_ids = np.argpartition(np.array(ucb), n_init)[
            :n_init].tolist()
        return [random_configs[i] for i in top_config_ids]
