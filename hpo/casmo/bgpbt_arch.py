from hpo.casmo.bgpbt import BGPBT
from hpo.casmo.casmo import Casmo4RL
import pandas as pd
import logging
from hpo.utils import get_reward_from_trajectory, is_large
import numpy as np
import ConfigSpace as CS
import shutil
import os
from copy import deepcopy
import random
from search_spaces.brax_env import Brax


class BGPBTArch(BGPBT):
    def __init__(
        self,
        env,
        log_dir,
        max_timesteps: int = None,
        pop_size: int = 4,
        n_init: int = None,
        verbose: bool = False,
        ard=False,
        t_ready: int = None,
        n_distillation_timesteps: int = int(5e5),
        quantile_fraction: float = 0.25,
        seed: int = None,
        use_reward_fraction: float = 0.0,
        existing_policy: str = "resume",
        init_policy: str = "bo",
        backtrack: bool = False,
        distill_every: int = int(3e6),
        patience: int = 15,
        max_distillation: int = 2,
        t_ready_end=None,
    ):
        """
        BGPBT with architecture search and distillation (i.e. the full method)
        Args:
            env: environment. Should be an object of the search space.
            log_dir: path str. The directory of the log & model saves
            pop_size: population size of BGPBT. The paper uses 8 by default.
                Note: int(pop_size * quantile_fraction) > 1 is required. Otherwise an error will be raised.
            n_init: int. Number of initializing population. If a value of n_init > pop_size is used, at initialization (and subsequent distillation)
            verbose: bool. Verbose mode
            ard: bool. Whether to enable automatic relevance determination in GP
            t_ready: int. Number of timestep intervals to run PBT iterations.
            n_distillation_timesteps: int. Number of timesteps to run per distillation. Note that due to the rollout steps, sometimes it may not be
                possible to control the exact of timesteps of distillation.
            quantile_fraction: float. The fraction of agent to be replaced at each iteration. If we have pop_size = 8 and quantile_fraction = 0.25,
                the bottom-performing 2 agents will be replaced by the top-performing 2 at each PBT iteration.
            seed: random seed.
            use_reward_fraction: how to estimate the reward trajectory. Default is that it will only use the last reward in the interval.
            existing_policy: str. options=["resume", "overwrite"] -- what to do when we have a log_dir with existing files (mostly due to a previous run).
            init_policy: "bo", "random": how to generate the initializing population after a distillation.
                Note that the population at the very beginning are chosen randomly regardless of how this keyword is set.
            distill_every: the maximum timestep interval before a distillation is triggered. Note that distillation may happen earlier if the reward has
                not been improving for ``patience`` number of PBT iterations.
            patience: the maximum number of PBT iterations that the algorithm will tolerate before triggering a new generation of distillation.
            max_distillation: Optional[int]. Maximum number of distillations/generations. When the number of distillations exceed this value, no new
                generation will be started, regardless of the settings of ``patiance`` and ``distil_every`.
            t_ready_end: Optional[int]. When set, BGPBT will anneal the t_ready from the start value of ``t_ready`` to ``t_ready_end`` in a linear way.
                When not provided, BGPBT will run a flat t_ready based on ``t_ready`` argment for the entire duration of training.
        """
        super(BGPBTArch, self).__init__(
            env,
            log_dir,
            max_timesteps,
            pop_size,
            n_init,
            verbose,
            ard,
            t_ready,
            quantile_fraction,
            seed,
            use_reward_fraction,
            existing_policy,
            backtrack=backtrack,
            schedule_t_ready=False,
            t_ready_end=t_ready_end,
        )
        self.n_distillation_timesteps = n_distillation_timesteps
        self.distill_every = distill_every
        self.patience = patience
        if self.resumed:
            self.last_distill_timestep = self.df[self.df.n_distills == self.n_distills][
                self.budget_type
            ].min()
            self.best_loss = self.df[self.df.n_distills ==
                                     self.n_distills].R.min()
            best_t = self.df[
                (self.df.n_distills == self.n_distills) *
                (self.df.R == self.best_loss)
            ]["t"].iloc[-1]
            self.n_fail = self.df.t.max() - best_t
        else:
            self.last_distill_timestep = 0
            self.n_fail = 0
            self.best_loss = float("inf")

        # process the NAS dimensions and the HPO dimensions
        self.nas_dims, self.nas_dim_names = [], []
        for i, dim in enumerate(self.env.config_space):
            if "NAS" in dim:
                self.nas_dims.append(
                    self.env.config_space.get_idx_by_hyperparameter_name(dim)
                )
                self.nas_dim_names.append(dim)
        assert init_policy in [
            "random",
            "bo",
        ], f"init_policy={init_policy} is invalid!"
        self.init_policy = init_policy
        self.best_config = None
        self.best_archs = None  # store the best archs at each stage
        self.max_distillation = max_distillation
        self.env.config_space.seed(self.seed)

    def search_init(self, best_agents=None):
        """Used to search the initializing population"""
        if self.n_distills > 0:  # distillation stages
            if best_agents is None or len(best_agents) == 0:
                raise ValueError()

        def f(
            configs,
            num_timesteps=None,
            ckpt_paths=None,
            teacher_configs=None,
            replace_teacher=False,
        ):
            if ckpt_paths is None:
                ckpt_paths = [
                    os.path.join(
                        f"{self.log_dir}/pb2_checkpoints",
                        f"{self.env.env_name}_InitConfig{i}_Stage{self.n_distills}.pt",
                    )
                    for i in range(0, len(configs))
                ]
            if num_timesteps is None:
                num_timesteps = self.t_ready_start
            n_large_models = sum([is_large(c["config"])
                                 for c in self.pop.values()])
            max_parallel = (
                self.env.max_parallel // 2
                if n_large_models / len(configs) >= 0.25
                else self.env.max_parallel
            )
            logging.info(
                f"Running config={configs} with n_parallel={max_parallel}")

            if self.n_distills == 0:  # search init for the very beginning
                trajectories = self.env.train_batch(
                    configs=configs,
                    seeds=[self.seed] * len(configs),
                    nums_timesteps=[num_timesteps] * len(configs),
                    max_parallel=max_parallel,
                    checkpoint_paths=ckpt_paths,
                )
            else:  # search init for the beginning of each distillation stage

                # run successive halving here
                # see notation in successive halving paper
                assert (
                    teacher_configs is not None and ckpt_paths is not None
                ), "For distillation, teacher_configs and teacher_ckpts must be specified!"
                trajectories = self.env.distill_batch(
                    teacher_configs=teacher_configs,
                    student_configs=configs,
                    seeds=[self.seed] * len(configs),
                    distill_nums_timesteps=[num_timesteps] * len(configs),
                    distill_total_num_timesteps=self.n_distillation_timesteps,
                    train_nums_timesteps=[0] * len(configs),
                    checkpoint_paths=ckpt_paths,
                    max_parallel=max(1, max_parallel // 2),
                    replace_teacher=replace_teacher,
                )
            return trajectories

        init_size = max(self.n_init, self.pop_size)
        if self.init_policy == "random" or self.n_distills == 0:
            # at initialization, randomly generate the configs for the architecture
            init_configs = [
                self.env.config_space.sample_configuration() for _ in range(init_size)
            ]
            # best_agents = []
        elif self.init_policy == "bo":  # use a BO for subsequent distillation steps
            # if using BO, for distillation we optimize for the arch only
            data = self.df[["Agent", "t", self.budget_type, "R", "n_distills"]]
            # whether to constrain the data to the current n_distills
            data = data[
                data["n_distills"] <= self.n_distills - 1
            ]  # get the data from the previous stage #
            data = data.dropna()
            # the configuration array
            data[["x{}".format(i) for i in range(len(self.df.conf[0]))]] = pd.DataFrame(
                self.df.conf.tolist(), index=self.df.index
            )
            archs = data[["x{}".format(i) for i in self.nas_dims] + ["R"]]
            best_perf_each_arch = (
                archs.groupby(["x{}".format(i) for i in self.nas_dims])
                .min()
                .reset_index()
            )
            # the last column is the return
            best_arch = (
                best_perf_each_arch[
                    best_perf_each_arch.R == best_perf_each_arch.R.min()
                ]
                .iloc[0, :-1]
                .values
            )
            nas_env = Brax(
                env_name=self.env.env_name,
                log_dir=self.log_dir,
                do_nas=True,
                do_hpo=False,
                seed=self.seed,
                use_categorical=self.env.use_categorical,
            )

            init_bo = Casmo4RL(
                env=nas_env, log_dir=self.log_dir, max_iters=100
            )  # dummy value
            init_bo._X = best_perf_each_arch.iloc[:, :-1].values
            init_bo._fX = best_perf_each_arch.iloc[:, -1].values
            # Fill half of the population with BO...
            suggested_archs = init_bo.suggest(
                n_suggestions=max(1, init_size // 2))
            # and the rest with randomly sampled archs
            len_suggested_archs = len(suggested_archs)
            if self.best_archs is None:
                self.best_archs = best_arch.reshape(1, -1)
            else:
                self.best_archs = np.unique(
                    np.concatenate((self.best_archs, best_arch.reshape(1, -1))), axis=0
                )

            # also add the best configs at each of the previous n_distill into the pool of suggested archs
            suggested_archs = np.concatenate(
                (suggested_archs, self.best_archs))
            if len_suggested_archs < init_size:
                for _ in range(len_suggested_archs, init_size):
                    random_arch = (
                        self.env.config_space.sample_configuration()
                        .get_array()[self.nas_dims]
                        .reshape(1, -1)
                    )
                    suggested_archs = np.concatenate(
                        (suggested_archs, random_arch))

            init_configs = []
            for i, suggested_arch in enumerate(suggested_archs):
                if i < len_suggested_archs:
                    # for distillation, we use the default hyperparams for the HPO dimensions.
                    init_config = deepcopy(
                        self.pop[random.choice(
                            np.arange(self.pop_size))]["config"]
                    ).get_array()
                else:
                    init_config = (
                        self.env.config_space.sample_configuration().get_array()
                    )
                init_config[self.nas_dims] = suggested_arch
                init_configs.append(
                    CS.Configuration(self.env.config_space, vector=init_config)
                )  # convert back to a CS.Configuration object.
        else:
            raise NotImplementedError()

        if (
            self.n_distills == 0
        ):  # if doing the search_init at the very beginning, simply copy the best weights over
            # run a larger size for a short number of steps to obtain low-fidelity proxies
            trajectories = f(init_configs)
            costs = [
                -get_reward_from_trajectory(
                    np.array(t["y"], dtype=np.float), use_last_fraction=self.use_reward
                )
                for t in trajectories
            ]
            rl_rewards = [
                get_reward_from_trajectory(np.array(t["y"], dtype=np.float), 0)
                for t in trajectories
            ]

            # get the best `self.pop_size' configs as the starting population
            top_config_ids = np.argsort(costs).tolist()

            for i, (agent, stats) in enumerate(self.pop.items()):
                self.pop[agent] = {
                    "done": False,
                    "config": init_configs[top_config_ids[i]],
                    "path": os.path.join(
                        f"{self.log_dir}/pb2_checkpoints",
                        f"{self.env.env_name}_seed{self.env.seed}_Agent{agent}.pt",
                    ),
                    "config_source": "random",
                    "distill": True,  # signals that this should be distilled in the next iteration
                }

                shutil.copy(
                    os.path.join(
                        f"{self.log_dir}/pb2_checkpoints",
                        f"{self.env.env_name}_InitConfig{top_config_ids[i]}_Stage{self.n_distills}.pt",
                    ),
                    os.path.join(
                        f"{self.log_dir}/pb2_checkpoints",
                        f"{self.env.env_name}_seed{self.env.seed}_Agent{agent}.pt",
                    ),
                )
            # delete the initialization checkpoints
            for i in range(len(top_config_ids)):
                os.remove(
                    os.path.join(
                        f"{self.log_dir}/pb2_checkpoints",
                        f"{self.env.env_name}_InitConfig{top_config_ids[i]}_Stage{self.n_distills}.pt",
                    )
                )

            current_t = 1
            for i in range(len(init_configs)):
                config = init_configs[i]
                config_array = config.get_array()
                rl_reward = rl_rewards[i]
                scalar_steps = trajectories[i]["x"][-1] + \
                    self.last_distill_timestep
                d = pd.DataFrame(columns=self.df.columns)
                agent_number = top_config_ids.index(i)
                if agent_number >= self.pop_size:
                    agent_number = -1
                # agent_number = -1 if (i not in top_config_ids) or (i >= self.pop_size) else
                path = self.pop[agent_number]["path"] if agent_number >= 0 else np.nan
                d.loc[0] = [
                    agent_number,
                    current_t,
                    scalar_steps,
                    costs[i],
                    rl_reward,
                    config_array.tolist(),
                    path,
                    config,
                    "random",
                    False,
                    self.n_distills,
                    np.nan,
                    np.nan,
                ]
                self.df = pd.concat([self.df, d]).reset_index(drop=True)
                logging.info(
                    "\nAgent: {}, Timesteps: {}, Cost: {}\n".format(
                        agent_number,
                        scalar_steps,
                        costs[i],
                    )
                )

        else:  # if doing the search_init for the subsequent steps, also do the distillation here and return the final archs
            teacher_configs, teacher_ckpts = [], []
            for i in range(len(init_configs)):
                best_agent = np.random.choice(best_agents)
                teacher_ckpt = f"{self.pop[best_agent]['path']}_forDistillAgent{i}"
                shutil.copy(self.pop[best_agent]["path"], teacher_ckpt)
                teacher_configs.append(self.pop[best_agent]["config"])
                teacher_ckpts.append(teacher_ckpt)
            logging.info(
                f"run_init distill student config={init_configs}. teacher configs = {teacher_configs}"
            )
            best_configs_for_distill = deepcopy(init_configs)

            # here we run successive halving to determine the top-'self.pop_size' configs for the next stage.
            distill_ckpts = deepcopy(teacher_ckpts)
            s = int(np.ceil(np.log(len(init_configs)) / np.log(self.pop_size)))
            eta = 2.0  # halving by default -- set anything above 2 for more aggressive elimination
            distill_timestep = 0

            elapsed_timestep = [0] * len(teacher_configs)
            for rung in range(s):
                if rung < s - 1:
                    timesteps_this_rung = int(
                        self.n_distillation_timesteps * eta ** (rung - s)
                    )  # see SuccessiveHalving paper
                else:
                    timesteps_this_rung = int(
                        self.n_distillation_timesteps - distill_timestep
                    )  # for the final rung, simply use up a
                logging.info(
                    f"Running SuccessiveHalving Rung={rung + 1}/{s}. Budgeted timestep={timesteps_this_rung}. "
                    f"Number of configs surviving in this rung={len(best_configs_for_distill)}"
                )
                trajs = f(
                    best_configs_for_distill,
                    num_timesteps=timesteps_this_rung,
                    teacher_configs=teacher_configs,
                    ckpt_paths=distill_ckpts,
                    replace_teacher=rung >= s - 1,
                )  # replace teacher at the final iter.
                for j, t in enumerate(trajs):
                    trajs[j]["x"] = (np.array(t["x"]) +
                                     elapsed_timestep[j]).tolist()

                costs = [
                    -get_reward_from_trajectory(
                        trajectory=t["y"], use_last_fraction=self.use_reward
                    )
                    for t in trajs
                ]
                distill_timestep = max([t["x"][-1] for t in trajs])
                ranked_reward_indices = np.argsort(costs)
                survived_agent_idices = ranked_reward_indices[
                    : max(self.pop_size, int(round(len(costs) / eta)))
                ]
                best_configs_for_distill = [
                    best_configs_for_distill[j] for j in survived_agent_idices
                ]
                teacher_configs = [teacher_configs[j]
                                   for j in survived_agent_idices]
                distill_ckpts = [distill_ckpts[j]
                                 for j in survived_agent_idices]
                elapsed_timestep = [elapsed_timestep[j]
                                    for j in survived_agent_idices]
                logging.info(
                    f"Surviving indices={survived_agent_idices}. "
                    f"Student configs={best_configs_for_distill}. Ckpts={distill_ckpts}"
                )

            full_trajectories = trajs
            # update self.pop with the new configs
            # the NAS dimensions of the new configs come from the previously identified best archs for this iteration,
            # and the HPO dimensions are a mix of randomly sampled and perturbed configs from the best config's HPO
            # dimensions before distillation.
            new_pop = deepcopy(self.pop)
            current_t = self.df.t.max() + 1
            for idx, (agent, stats) in enumerate(new_pop.items()):
                # retain the values of the NAS dimensions
                logging.info(
                    f"Assigning {best_configs_for_distill[idx]} stored at {distill_ckpts[idx]} to Agent {agent}"
                )
                shutil.copy(distill_ckpts[idx], self.pop[agent]["path"])
                new_config = deepcopy(best_configs_for_distill[idx])
                new_pop[agent] = {
                    "done": False,
                    "config": new_config,
                    "path": self.pop[agent]["path"],
                    "config_source": "distilled",
                    "distill": False,  # signals that this should be distilled in the next iteration
                }

                # record the rewards in self.df
                d = pd.DataFrame(columns=self.df.columns)
                rl_reward = get_reward_from_trajectory(
                    np.array(full_trajectories[agent]["y"]), 0
                )
                cost = -get_reward_from_trajectory(
                    np.array(full_trajectories[agent]["y"]),
                    use_last_fraction=self.use_reward,
                )
                max_t = self.df[self.df.Agent == agent][self.budget_type].max()
                scalar_steps = full_trajectories[agent]["x"][-1] + max_t
                d.loc[0] = [
                    agent,
                    current_t,
                    scalar_steps,
                    cost,
                    rl_reward,
                    new_config.get_array().tolist(),
                    self.pop[agent]["path"],
                    new_config,
                    "random",
                    False,
                    self.n_distills,
                    np.nan,
                    np.nan,
                ]
                self.df = pd.concat([self.df, d]).reset_index(drop=True)
            self.pop = new_pop

            # remove the temporary checkpoints used during distillation
            for teacher_ckpt in teacher_ckpts:
                os.remove(teacher_ckpt)

    def run(self):
        all_done = False
        distill_at_this_step = False
        if not self.resumed:
            logging.info("Searching for initialising configs!")
            self.search_init()

        while not all_done:
            # to avoid GPU OOM
            n_large_models = sum([is_large(c["config"])
                                 for c in self.pop.values()])
            max_parallel = (
                self.env.max_parallel // 2
                if n_large_models / len(self.pop) >= 0.25
                else self.env.max_parallel
            )

            if distill_at_this_step:
                n = max(int(self.pop_size * self.quantile_fraction), 1)
                logging.info(
                    f"Modifying the nets at this iteration & distilling.")

                last_entries = self.df[
                    self.df["t"] == self.df.t.max()
                ]  # index entire population based on last set of runs
                last_entries = last_entries.iloc[
                    : self.pop_size
                ]  # only want the original entries
                ranked_last_entries = last_entries.sort_values(
                    by=["R"], ignore_index=True, ascending=False
                )  # rank last entries
                best_agents = list(
                    ranked_last_entries.iloc[-n:]["Agent"].values)
                # clone the best agents weights and their params
                best_agent_configs_prev = {}

                self.search_init(best_agents=best_agents)
                distill_at_this_step = False

            logging.info(
                f"Max parallel for this iteration={max_parallel}. Last distillation step={self.last_distill_timestep}"
            )
            logging.info(
                f'Running config={[c["config"] for c in self.pop.values()]}')

            # Whether use an annealing schedule of t_ready
            if self.t_ready_end == self.t_ready_start or self.t_ready_end is None:
                t_ready = self.t_ready_start
            else:
                t_ready = int(
                    self.t_ready_start
                    + (self.t_ready_end - self.t_ready_start)
                    / self.max_timesteps
                    * self.df[self.budget_type].max()
                )

            results_values = self.env.train_batch(
                configs=[c["config"] for c in self.pop.values()],
                seeds=[self.seed] * len(self.pop),
                nums_timesteps=[t_ready] * len(self.pop),
                checkpoint_paths=[c["path"] for c in self.pop.values()],
                max_parallel=max_parallel,
            )

            results_keys = list(self.pop.keys())
            results = dict(zip(results_keys, results_values))

            if self.df.t.empty:
                t = 1
            else:
                t = self.df.t.max() + 1

            for agent in self.pop.keys():
                # negative sign to convert the reward maximization to a minimisation problem
                self.pop[agent]["distill"] = False
                if self.pop[agent]["done"]:
                    logging.info(f"Skipping completed agent {agent}.")
                    continue

                final_cost = -get_reward_from_trajectory(
                    results[agent]["y"], self.use_reward, 0.0
                )
                rl_reward = get_reward_from_trajectory(
                    results[agent]["y"], 1, 0.0)
                final_timestep = results[agent]["x"][-1]

                if self.df[self.df["Agent"] == agent].empty:
                    scalar_steps = final_timestep
                else:
                    scalar_steps = (
                        final_timestep
                        + self.df[self.df["Agent"] ==
                                  agent][self.budget_type].max()
                    )
                logging.info(
                    "\nAgent: {}, Timesteps: {}, Cost: {}\n".format(
                        agent, scalar_steps, final_cost
                    )
                )

                conf_array = self.pop[agent]["config"].get_array().tolist()
                conf = self.pop[agent]["config"]
                config_source = self.pop[agent]["config_source"]
                d = pd.DataFrame(columns=self.df.columns)
                d.loc[0] = [
                    agent,
                    t,
                    scalar_steps,
                    final_cost,
                    rl_reward,
                    conf_array,
                    self.pop[agent]["path"],
                    conf,
                    config_source,
                    False,
                    self.n_distills,
                    self.policy_net,
                    self.value_net,
                ]
                self.df = pd.concat([self.df, d]).reset_index(drop=True)

                if (
                    self.df[self.df["Agent"] == agent][self.budget_type].max()
                    >= self.max_timesteps
                ):
                    self.pop[agent]["done"] = True

                # update the trust region based on the results of the agents from previous runs, before exploitation
            self.adjust_tr_length()

            best_loss = self.df[self.df.n_distills ==
                                self.n_distills]["R"].min()
            if self.backtrack:
                if best_loss < self.best_cost:
                    self.best_cost = best_loss
                    overall_best_agent = self.df[
                        (self.df["R"] == best_loss)
                        & (self.df["n_distills"] == self.n_distills)
                    ].iloc[-1]
                    shutil.copy(
                        overall_best_agent["path"], self.best_checkpoint_dir)
                    self.best_config = deepcopy(overall_best_agent["conf_"])

            # exploitation -- copy the weights and etc.
            for agent in self.pop.keys():
                old_conf = self.pop[agent]["config"].get_array().tolist()
                self.pop[agent], copied = self.exploit(
                    agent,
                )
                # here we need to include a way to account for changes in the data.
                new_conf = self.pop[agent]["config"].get_array().tolist()
                if not np.isclose(
                    0, np.nansum(np.array(old_conf) - np.array(new_conf))
                ):
                    logging.info("changing conf for agent: {}".format(agent))
                    new_row = self.df[
                        (self.df["Agent"] == copied) & (
                            self.df["t"] == self.df.t.max())
                    ]
                    new_row["Agent"] = agent
                    # new_row['path'] = self.pop[agent]['path']
                    logging.info(f"new row conf old: {new_row['conf']}")
                    logging.info(f"new row conf new: {[new_conf]}")
                    new_row["conf"] = [new_conf]
                    new_row["conf_"] = [
                        CS.Configuration(
                            self.env.config_space, vector=new_conf)
                    ]
                    self.df = pd.concat(
                        [self.df, new_row]).reset_index(drop=True)
                    logging.info(f"new config: {new_conf}")

            all_done = np.array(
                [self.pop[agent]["done"] for agent in self.pop.keys()]
            ).all()
            # save intermediate results
            self.df.to_csv(
                os.path.join(
                    self.log_dir, f"stats_seed_{self.seed}_intermediate.csv")
            )
            self.last_distill_timestep = self.df[self.df.n_distills == self.n_distills][
                self.budget_type
            ].min()  # record the timestep as the last time we undergo distillation.
            t_max = self.df[self.budget_type].max()
            best_loss = self.df[self.df["n_distills"]
                                == self.n_distills].R.min()
            if (
                self.df[
                    (self.df[self.budget_type] == t_max)
                    & (self.df["n_distills"] == self.n_distills)
                ].R.min()
                == best_loss
            ):
                self.n_fail = 0
            else:
                self.n_fail += 1
            # restart when the casmo trust region is below threshold
            if self.n_distills < self.max_distillation and (
                self.n_fail >= self.patience
                or t_max - self.last_distill_timestep > self.distill_every
            ):
                distill_at_this_step = True
                self.n_distills += 1
                self.n_fail = 0
                self.best_cost = float("inf")
                logging.info("Start distillation in the next iteration..")
            logging.info(f"n_fail: {self.n_fail}")
        return self.df

    def explore(self, agent, best_agent):
        """Fit a Casmo model to the existing data and run BayesOpt to suggest new configurations."""
        dfnewpoint, data, agent_t = self.format_df(agent, best_agent)
        # dfnewpoint contains the information of the previous best agent. At the explore step, we replace it with an
        #   alternative hparam config.
        # find the config of the best agents and replace the NAS dimensions
        if self.backtrack:
            best_agent_configs_array = self.best_config.get_array()
        else:
            best_agent_configs_array = self.pop[best_agent]["config"].get_array(
            )

        best_agent_nas_vals = best_agent_configs_array[self.nas_dims]
        if not dfnewpoint.empty:
            y = np.array(data.y.values)
            t = np.array(data.t.values)
            # hyperparameter dimensions -- active dimensions that will be tuned.
            hparams = data[["x{}".format(i)
                            for i in range(len(self.df.conf[0]))]]
            # fix the nas dimensions
            hparams[[f"x{i}" for i in self.nas_dims]] = best_agent_nas_vals

            # contextual dimension
            current = np.array(
                [x for x in self.running[str(agent_t)].values()])
            # concatenate the array with the contextual dimensions
            t_current = np.tile(
                np.array(dfnewpoint.t.values), current.shape[0])

            # it is important that the contextual information is appended to the end of the vector.
            t_r = data[
                ["R_before"]
            ]  # fixed dimensions -- serve as contextual information for BO but may not be modified.
            X = pd.concat([hparams, t_r], axis=1).values
            t_r_current = np.tile(
                dfnewpoint[["R_before"]].values, (current.shape[0], 1)
            )
            current = np.hstack([current, t_r_current]).astype(float)
            # get the hp of the best agent selected from -- this will be trust region centre
            X_best = dfnewpoint[
                ["x{}".format(i) for i in range(
                    len(self.df.conf[0]))] + ["R_before"]
            ].values

            new_config_array = self.select_config(
                X,
                y,
                t,
                current,
                t_current,
                x_center=X_best,
                frozen_dims=self.nas_dims,
                frozen_vals=best_agent_nas_vals,
            )
            new_config = CS.Configuration(
                self.env.config_space, vector=new_config_array
            )
            config_source = "bo"
        else:
            logging.info("Using random exploration.")
            new_config = self.env.config_space.sample_configuration()
            new_config_array = new_config.get_array()
            # note that the NAS dimensions are fixed
            new_config_array[self.nas_dims] = best_agent_nas_vals
            new_config = CS.Configuration(
                self.env.config_space, vector=new_config_array
            )
            config_source = "random"

        to_use = new_config.get_array().tolist()
        try:
            self.running[str(agent_t)].update({str(agent): to_use})
        except KeyError:
            self.running.update({str(agent_t): {str(agent): to_use}})
        return new_config, config_source
