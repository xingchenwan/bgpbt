import collections
import math
import time
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
import json
import pathlib
from custom_brax_train.utils import LinearSchedule
import logging
import copy

activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'silu': nn.SiLU(),
    'hardswish': nn.Hardswish(),
}


class Agent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 policy_hidden_layer_sizes: str,
                 policy_activation,
                 v_hidden_layer_sizes: str,
                 value_activation,
                 entropy_cost: float,
                 discounting: float,
                 reward_scaling: float,
                 lambda_: float,
                 ppo_epsilon: float,
                 unroll_length: int,
                 batch_size: int,
                 num_minibatches: int,
                 num_update_epochs: int,
                 device: str,
                 # schedule_lr: bool,
                 ):
        super(Agent, self).__init__()

        # TODO: Make networks changable with distillation
        if isinstance(policy_hidden_layer_sizes, str):
            policy_hidden_layer_sizes = [
                int(x) for x in policy_hidden_layer_sizes.split(',')]
        if isinstance(v_hidden_layer_sizes, str):
            v_hidden_layer_sizes = [int(x)
                                    for x in v_hidden_layer_sizes.split(',')]
        print(f'Policy Network: {policy_hidden_layer_sizes}')
        print(f'Value Network: {v_hidden_layer_sizes}')
        # Create the agent
        policy_layers = [obs_dim] + policy_hidden_layer_sizes + [act_dim * 2]
        value_layers = [obs_dim] + v_hidden_layer_sizes + [1]
        policy = []
        for w1, w2 in zip(policy_layers, policy_layers[1:]):
            policy.append(nn.Linear(w1, w2))
            policy.append(policy_activation)
        policy.pop()  # Drop the final activation
        self.policy = nn.Sequential(*policy)
        value = []
        for w1, w2 in zip(value_layers, value_layers[1:]):
            value.append(nn.Linear(w1, w2))
            value.append(value_activation)
        value.pop()  # Drop the final activation
        self.value = nn.Sequential(*value)

        # Norm parameters
        self.num_steps = torch.zeros((), device=device)
        self.running_mean = torch.zeros(obs_dim, device=device)
        self.running_variance = torch.zeros(obs_dim, device=device)

        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = lambda_
        self.ppo_epsilon = ppo_epsilon
        self.unroll_length = unroll_length
        self.batch_size = batch_size
        self.num_minibatches = num_minibatches
        self.num_update_epochs = num_update_epochs
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    @torch.jit.export
    def increment_size(self):
        return self.batch_size * self.num_minibatches * self.unroll_length

    @torch.jit.export
    def update_training_hyperparameters(
            self,
            discounting: Optional[float] = None,
            entropy_cost: Optional[float] = None,
            unroll_length: Optional[int] = None,
            batch_size: Optional[int] = None,
            num_minibatches: Optional[int] = None,
            num_update_epochs: Optional[int] = None,
            reward_scaling: Optional[float] = None,
            lambda_: Optional[float] = None,
            ppo_epsilon: Optional[float] = None,
    ):
        if discounting is not None:
            self.discounting = discounting
        if entropy_cost is not None:
            self.entropy_cost = entropy_cost
        if unroll_length is not None:
            self.unroll_length = unroll_length
        if batch_size is not None:
            self.batch_size = batch_size
        if num_minibatches is not None:
            self.num_minibatches = num_minibatches
        if num_update_epochs is not None:
            self.num_update_epochs = num_update_epochs
        if reward_scaling is not None:
            self.reward_scaling = reward_scaling
        if lambda_ is not None:
            self.lambda_ = lambda_
        if ppo_epsilon is not None:
            self.ppo_epsilon = ppo_epsilon

    @torch.jit.export
    def dist_create(self, logits):
        """Normal followed by tanh.

        torch.distribution doesn't work with torch.jit, so we roll our own."""
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + .001
        return loc, scale

    @torch.jit.export
    def dist_sample_no_postprocess(self, loc, scale):
        return torch.normal(loc, scale)

    @classmethod
    def dist_postprocess(cls, x):
        return torch.tanh(x)

    @torch.jit.export
    def dist_entropy(self, loc, scale):
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        entropy = 0.5 + log_normalized
        entropy = entropy * torch.ones_like(loc)
        # Bug fix for scale
        scale = torch.max(scale, torch.tensor(1e-3).float())
        dist = torch.normal(loc, scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        entropy = entropy + log_det_jacobian
        return entropy.sum(dim=-1)

    @torch.jit.export
    def dist_log_prob(self, loc, scale, dist):
        log_unnormalized = -0.5 * ((dist - loc) / scale).square()
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        log_prob = log_unnormalized - log_normalized - log_det_jacobian
        return log_prob.sum(dim=-1)

    @torch.jit.export
    def update_normalization(self, observation):
        self.num_steps += observation.shape[0] * observation.shape[1]
        input_to_old_mean = observation - self.running_mean
        mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        self.running_mean = self.running_mean + mean_diff
        input_to_new_mean = observation - self.running_mean
        var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        self.running_variance = self.running_variance + var_diff

    @torch.jit.export
    def normalize(self, observation):
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6)
        return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    @torch.jit.export
    def get_logits_action(self, observation):
        observation = self.normalize(observation)
        logits = self.policy(observation)
        loc, scale = self.dist_create(logits)
        action = self.dist_sample_no_postprocess(loc, scale)
        return logits, action

    @torch.jit.export
    def compute_gae(self, truncation, termination, reward, values,
                    bootstrap_value):
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
        deltas = reward + self.discounting * (
            1 - termination) * values_t_plus_1 - values
        deltas *= truncation_mask

        acc = torch.zeros_like(bootstrap_value)
        vs_minus_v_xs = torch.zeros_like(truncation_mask)

        for ti in range(truncation_mask.shape[0]):
            ti = truncation_mask.shape[0] - ti - 1
            acc = deltas[ti] + self.discounting * (
                1 - termination[ti]) * truncation_mask[ti] * self.lambda_ * acc
            vs_minus_v_xs[ti] = acc

        # Add V(x_s) to get v_s.
        vs = vs_minus_v_xs + values
        vs_t_plus_1 = torch.cat(
            [vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
        advantages = (reward + self.discounting *
                      (1 - termination) * vs_t_plus_1 - values) * truncation_mask
        return vs, advantages

    @torch.jit.export
    def compute_policy_and_value(self, td: Dict[str, torch.Tensor]):
        observation = self.normalize(td['observation'])
        logits = self.policy(observation[:-1])
        loc, scale = self.dist_create(logits)

        baseline = self.value(observation)
        baseline = torch.squeeze(baseline, dim=-1)
        return baseline, loc, scale

    @torch.jit.export
    def loss(self, td: Dict[str, torch.Tensor]):
        observation = self.normalize(td['observation'])
        policy_logits = self.policy(observation[:-1])
        baseline = self.value(observation)
        baseline = torch.squeeze(baseline, dim=-1)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = baseline[-1]
        baseline = baseline[:-1]
        reward = td['reward'] * self.reward_scaling
        termination = td['done'] * (1 - td['truncation'])

        loc, scale = self.dist_create(td['logits'])
        behaviour_action_log_probs = self.dist_log_prob(
            loc, scale, td['action'])
        loc, scale = self.dist_create(policy_logits)
        target_action_log_probs = self.dist_log_prob(loc, scale, td['action'])

        with torch.no_grad():
            vs, advantages = self.compute_gae(
                truncation=td['truncation'],
                termination=termination,
                reward=reward,
                values=baseline,
                bootstrap_value=bootstrap_value)

        rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.ppo_epsilon,
                                     1 + self.ppo_epsilon) * advantages
        policy_loss = - \
            torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline
        v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

        # Entropy reward
        entropy = torch.mean(self.dist_entropy(loc, scale))
        entropy_loss = self.entropy_cost * -entropy

        return policy_loss + v_loss + entropy_loss


StepData = collections.namedtuple(
    'StepData',
    ('observation', 'logits', 'action', 'reward', 'done', 'truncation'))


def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
    """Map a function over each field in StepData."""
    items = {}
    keys = sds[0]._asdict().keys()
    for k in keys:
        items[k] = f(*[sd._asdict()[k] for sd in sds])
    return StepData(**items)


def eval_unroll(agent, env, length):
    """Return number of episodes and average reward for a single unroll."""
    observation = env.reset()
    episodes = torch.zeros((), device=agent.device)
    episode_reward = torch.zeros((), device=agent.device)
    for _ in range(length):
        _, action = agent.get_logits_action(observation)
        observation, reward, done, _ = env.step(Agent.dist_postprocess(action))
        episodes += torch.sum(done)
        episode_reward += torch.sum(reward)
    return episodes, episode_reward / episodes


def train_unroll(agent, env, observation, num_unrolls, unroll_length):
    """Return step data over multple unrolls."""
    sd = StepData([], [], [], [], [], [])
    for _ in range(num_unrolls):
        one_unroll = StepData([observation], [], [], [], [], [])
        for _ in range(unroll_length):
            logits, action = agent.get_logits_action(observation)
            observation, reward, done, info = env.step(
                Agent.dist_postprocess(action))
            one_unroll.observation.append(observation)
            one_unroll.logits.append(logits)
            one_unroll.action.append(action)
            one_unroll.reward.append(reward)
            one_unroll.done.append(done)
            one_unroll.truncation.append(info['truncation'])
        one_unroll = sd_map(torch.stack, one_unroll)
        sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
    td = sd_map(torch.stack, sd)
    return observation, td


# make unroll first
def unroll_first(data):
    data = data.swapaxes(0, 1)
    return data.reshape([data.shape[0], -1] + list(data.shape[3:]))


class PPOTrainer:
    def __init__(self,
                 env,
                 agent,
                 learning_rate: float,
                 save_path: pathlib.Path,
                 episode_length: int,
                 eval_every: int = int(1e6),
                 schedule_lr: bool = False,
                 ):
        self._env = env
        self.save_path = save_path
        self.episode_length = episode_length
        self.eval_every = eval_every
        self.observation = self._env.reset()
        self.sps = 0
        self.total_steps = 0
        self.total_loss = 0
        self.metrics = {
            'num_steps': [],
            'eval/episode_reward': [],
            'eval/completed_episodes': [],
            'eval/avg_episode_length': [],
            'speed/sps': [],
            'speed/eval_sps': [],
            'losses/total_loss': [],
        }

        # Hold agents in here and update them concurrently.
        self.teacher = {
            'agent': agent,
            'opt': torch.optim.Adam(agent.parameters(), lr=learning_rate),
        }
        self.students = []
        self.device = self.teacher['agent'].device

        # Distillation alpha scheduler
        self.alpha_schedule = None
        self.schedule_lr = schedule_lr

        # Initial evaluation
        self.evaluate()

    def evaluate(self, agent=None):
        if agent is None:
            agent = self.teacher['agent']
        t = time.time()
        with torch.no_grad():
            episode_count, episode_reward = eval_unroll(
                agent, self._env, self.episode_length)
        duration = time.time() - t

        episode_avg_length = self._env.num_envs * self.episode_length / episode_count
        eval_sps = self._env.num_envs * self.episode_length / duration
        progress = {
            'num_steps': self.total_steps,
            'eval/episode_reward': episode_reward,
            'eval/completed_episodes': episode_count,
            'eval/avg_episode_length': episode_avg_length,
            'speed/sps': self.sps,
            'speed/eval_sps': eval_sps,
            'losses/total_loss': self.total_loss,
        }
        print(progress)
        for k, v in progress.items():
            self.metrics[k].append(float(v))
        with open(self.save_path, 'w') as fp:
            json.dump(self.metrics, fp, indent=2)
        return progress

    def run(self, num_timesteps):
        num_steps = self.teacher['agent'].increment_size()
        log_frequency = max(num_timesteps // self.eval_every, 1)
        num_epochs = max(num_timesteps // (num_steps * log_frequency), 1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.teacher['opt'], eta_min=1e-8, T_max=num_epochs * log_frequency) if self.schedule_lr else None
        print(f'LR scheduling = {lr_scheduler is not None}')

        for _ in range(log_frequency):
            self.observation = self._env.reset()
            t = time.time()
            self.total_loss = 0

            # Run epochs of training
            for _ in range(num_epochs):
                td = self.unroll_agent(self.teacher['agent'])
                # Update teacher
                for _ in range(self.teacher['agent'].num_update_epochs):
                    for td_minibatch in self.shuffle_and_batch(td, self.teacher['agent'].num_minibatches):
                        loss = self.teacher['agent'].loss(
                            td_minibatch._asdict())
                        self.teacher['opt'].zero_grad()
                        loss.backward()
                        self.teacher['opt'].step()
                        self.total_loss += loss.detach()
                self.total_steps += num_steps
                if lr_scheduler is not None:
                    lr_scheduler.step()
                    logging.info(
                        f'Current learning rate = {lr_scheduler.get_last_lr()}')

            duration = time.time() - t
            self.total_loss = self.total_loss / \
                (num_epochs * self.teacher['agent'].num_update_epochs *
                 self.teacher['agent'].num_minibatches)
            self.sps = num_epochs * num_steps / duration
            self.evaluate()

    def unroll_agent(self, agent):
        num_unrolls = agent.batch_size * agent.num_minibatches // self._env.num_envs
        self.observation, td = train_unroll(agent, self._env, self.observation, num_unrolls,
                                            agent.unroll_length)
        td = sd_map(unroll_first, td)
        agent.update_normalization(td.observation)
        return td

    def shuffle_and_batch(self, td, num_minibatches):
        with torch.no_grad():
            permutation = torch.randperm(
                td.observation.shape[1], device=self.device)

            def shuffle_batch(data):
                data = data[:, permutation]
                data = data.reshape([data.shape[0], num_minibatches, -1] +
                                    list(data.shape[2:]))
                return data.swapaxes(0, 1)

            epoch_td = sd_map(shuffle_batch, td)
        for minibatch_i in range(num_minibatches):
            yield sd_map(lambda d: d[minibatch_i], epoch_td)

    def set_distillation_schedule(
            self,
            total_timesteps,
            distill_anneal_frac: float = 0.8,
            distill_anneal_init: float = 1.,
            distill_anneal_final: float = 0.1,
    ):
        self.alpha_schedule = LinearSchedule(
            burnin=self.total_steps,
            initial_value=distill_anneal_init,
            final_value=distill_anneal_final,
            decay_time=distill_anneal_frac * total_timesteps,
        )

    def distill(
            self,
            num_timesteps,
            policy_reg_coef: float = 1.,
            value_reg_coef: float = 0.5,
            rl_coef: float = 1.,
            distill_num_epochs: int = 4,
    ):
        assert self.alpha_schedule is not None, "Distillation schedule not set"
        # print(num_timesteps)
        for student_idx, student in enumerate(self.students):
            logging.info(
                f'Distilling student: {student_idx + 1}/{len(self.students)}')
            num_steps = student['agent'].increment_size()
            log_frequency = max(num_timesteps // self.eval_every, 1)
            num_epochs = max(num_timesteps // (num_steps * log_frequency), 1)
            for _ in range(log_frequency):
                self.observation = self._env.reset()
                t = time.time()
                self.total_loss = 0

                # Run epochs of training
                for _ in range(num_epochs):
                    td = self.unroll_agent(student['agent'])
                    # Update teacher
                    for _ in range(distill_num_epochs):
                        for td_minibatch in self.shuffle_and_batch(td, student['agent'].num_minibatches):
                            with torch.no_grad():
                                teacher_v, teacher_loc, teacher_scale = self.teacher['agent'].compute_policy_and_value(
                                    td_minibatch._asdict())

                            rl_loss = student['agent'].loss(
                                td_minibatch._asdict())
                            student_v, student_loc, student_scale = student['agent'].compute_policy_and_value(
                                td_minibatch._asdict())

                            # Value loss is L2 loss
                            val_loss = (student_v - teacher_v).pow(2).mean()
                            # # Policy loss is KL divergence
                            te = torch.distributions.Normal(
                                teacher_loc, teacher_scale)
                            st = torch.distributions.Normal(
                                student_loc, student_scale)
                            policy_loss = torch.distributions.kl_divergence(
                                te, st).mean()

                            alpha = self.alpha_schedule(self.total_steps)

                            distill_loss = rl_coef * rl_loss + alpha * (
                                value_reg_coef * val_loss + policy_reg_coef * policy_loss)

                            student['opt'].zero_grad()
                            distill_loss.backward()
                            student['opt'].step()
                            self.total_loss += distill_loss.detach()
                    self.total_steps += num_steps

                duration = time.time() - t
                self.total_loss = self.total_loss / \
                    (num_epochs * distill_num_epochs *
                     student['agent'].num_minibatches)
                self.sps = num_epochs * num_steps / duration
                self.evaluate(student['agent'])

    def attach_student(self, params):
        logging.info(f'Attaching student!')
        new_agent = Agent(
            obs_dim=self.teacher['agent'].obs_dim,
            act_dim=self.teacher['agent'].act_dim,
            policy_hidden_layer_sizes=params['policy_hidden_layer_sizes'],
            policy_activation=activations[params['policy_activation']],
            v_hidden_layer_sizes=params['v_hidden_layer_sizes'],
            value_activation=activations[params['v_activation']],
            entropy_cost=params['entropy_cost'],
            discounting=params['discounting'],
            reward_scaling=params['reward_scaling'],
            lambda_=params['lambda_'],
            ppo_epsilon=params['ppo_epsilon'],
            unroll_length=params['unroll_length'],
            batch_size=params['batch_size'],
            num_minibatches=params['num_minibatches'],
            num_update_epochs=params['num_update_epochs'],
            device=self.device,
        )
        new_agent = torch.jit.script(new_agent.to(self.device))
        self.students.append({
            'agent': new_agent,
            'opt': torch.optim.Adam(new_agent.parameters(), lr=params['learning_rate']),
            'params': copy.deepcopy(params),
        })

    def replace_teacher(self, idx=None):
        logging.info(f'Replacing teacher with student!')
        del self.teacher
        if idx is None:
            self.teacher = self.students[0]
        else:
            self.teacher = self.students[idx]
        self.students = []

    def update_training_hyperparameters(
            self,
            discounting: Optional[float] = None,
            entropy_cost: Optional[float] = None,
            unroll_length: Optional[int] = None,
            batch_size: Optional[int] = None,
            learning_rate: Optional[float] = None,
            num_minibatches: Optional[int] = None,
            num_update_epochs: Optional[int] = None,
            reward_scaling: Optional[float] = None,
            lambda_: Optional[float] = None,
            ppo_epsilon: Optional[float] = None,
            student_idx: Optional[int] = None,
            **kwargs
    ):
        if student_idx is None:
            updated = self.teacher
        else:
            updated = self.students[student_idx]

        if learning_rate is not None:
            updated['opt'].param_groups[0]['lr'] = learning_rate
        updated['agent'].update_training_hyperparameters(
            discounting=discounting,
            entropy_cost=entropy_cost,
            unroll_length=unroll_length,
            batch_size=batch_size,
            num_minibatches=num_minibatches,
            num_update_epochs=num_update_epochs,
            reward_scaling=reward_scaling,
            lambda_=lambda_,
            ppo_epsilon=ppo_epsilon,
        )

    def save_checkpoint(self, path):
        state = {
            'agent': self.teacher['agent'].state_dict(),
            'opt': self.teacher['opt'].state_dict(),
            'num_steps': self.teacher['agent'].num_steps,
            'running_mean': self.teacher['agent'].running_mean,
            'running_variance': self.teacher['agent'].running_variance,
            'students': [
                {'agent': s['agent'].state_dict(),
                 'opt': s['opt'].state_dict(),
                 'num_steps': s['agent'].num_steps,
                 'running_mean': s['agent'].running_mean,
                 'running_variance': s['agent'].running_variance,
                 'params': s['params']} for s in self.students
            ],
            'alpha_schedule': self.alpha_schedule,
        }
        torch.save(state, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.teacher['agent'].load_state_dict(checkpoint['agent'])
        self.teacher['opt'].load_state_dict(checkpoint['opt'])
        self.teacher['agent'].num_steps = checkpoint['num_steps']
        self.teacher['agent'].running_mean = checkpoint['running_mean']
        self.teacher['agent'].running_variance = checkpoint['running_variance']
        # Backwards compatible with old checkpoints without students.
        if 'students' in checkpoint:
            for s in checkpoint['students']:
                self.attach_student(s['params'])
                self.students[-1]['agent'].load_state_dict(s['agent'])
                self.students[-1]['opt'].load_state_dict(s['opt'])
                self.students[-1]['agent'].num_steps = s['num_steps']
                self.students[-1]['agent'].running_mean = s['running_mean']
                self.students[-1]['agent'].running_variance = s['running_variance']
        if 'alpha_schedule' in checkpoint:
            self.alpha_schedule = checkpoint['alpha_schedule']
