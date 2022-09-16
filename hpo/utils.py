from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from hpo.casmo.kernels import CasmoKernel
from gpytorch.constraints.constraints import Interval
import numpy as np
from gpytorch.likelihoods import GaussianLikelihood
import ConfigSpace as CS
from ConfigSpace.util import deactivate_inactive_hyperparameters
import ConfigSpace.hyperparameters as CSH
import torch
from copy import deepcopy
from typing import Callable
import logging
import scipy.stats as ss
from typing import List


def is_large(config: CS.Configuration):
    """Roughly estimate the memory burden of a config to avoid OOM errors."""
    config_dict = config.get_dictionary()
    if 'unroll_length' in config_dict.keys() and config_dict['unroll_length'] >= 15:
        return True
    if 'NAS_policy_log2_width' in config_dict.keys() and config_dict['NAS_policy_log2_width'] >= 8 \
            and 'NAS_policy_num_layers' in config_dict.keys() and config_dict['NAS_policy_num_layers'] >= 4:
        # models larger than (256, 256, 256, 256)
        return True
    if 'NAS_policy_log2_width' in config_dict.keys() and config_dict['NAS_policy_log2_width'] >= 8 \
            and 'NAS_policy_num_layers' in config_dict.keys() and config_dict['NAS_policy_num_layers'] >= 4:
        # models larger than (256, 256, 256, 256)
        return True
    return False


def get_reward_from_trajectory(trajectory: np.array, use_last_fraction: float, risk_aversion: float = 0.):
    """
    Compute the final reward from a trajectory, based on drawdown (intead of standard deviation).
    use_last_fraction: 0 < .. < 1, the final reward will be computed as the simple average of the rewards.
    risk_aversion: (not currently used). Whether to penalize instabilities by considering the maximum drawdown
        from the max of the reward trajectory.
    """
    if not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory).astype(np.float)
    if use_last_fraction == 1:
        use_final_n = 1     # 1 is interpreted as using the last entry only
    else:
        use_final_n = max(
            1, int(np.round(use_last_fraction * len(trajectory))))
    rew = np.median(trajectory[-use_final_n:])
    if np.abs(risk_aversion) > 1e-3:
        rolling_max = np.array([0] + [np.max(trajectory[:i])
                               for i in range(1, len(trajectory + 1))]).astype(np.float)
        max_drawdown = max(rolling_max - trajectory)
    else:
        max_drawdown = 0.
    return rew - risk_aversion * max_drawdown


# GP Model

def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats


def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    # random.seed(random.randint(0, 1e6))
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X


class GP(ExactGP):
    def __init__(self, train_x, train_y, kern, likelihood,
                 outputscale_constraint, ):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.dim = train_x.shape[1]
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            kern, outputscale_constraint=outputscale_constraint)

    def forward(self, x, x_mask=None, ):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x1_mask=x_mask, )
        return MultivariateNormal(mean_x, covar_x)


def train_gp(configspace: CS.ConfigurationSpace,
             train_x, train_y, use_ard, num_steps,
             time_varying: bool = False,
             train_t=None,
             lengthscale_scaling: float = 2.,
             hypers={},
             noise_variance=None,
             return_hypers=False,
             verbose: bool = False
             ):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized.
    (train_x, train_y): pairs of x and y (trained)
    noise_variance: if provided, this value will be used as the noise variance for the GP model. Otherwise, the noise
        variance will be inferred from the model.

    """
    from math import sqrt
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]
    if train_t is not None:
        if not isinstance(train_t, torch.Tensor):
            train_t = torch.tensor(train_t).to(dtype=train_x.dtype)

    # Create hyper parameter bounds
    if noise_variance is None:
        noise_variance = 0.001
        noise_constraint = Interval(1e-6, 0.1)
    else:
        if np.abs(noise_variance) < 1e-6:
            noise_variance = 0.02
            noise_constraint = Interval(1e-6, 0.05)
        else:
            noise_constraint = Interval(
                0.99 * noise_variance, 1.01 * noise_variance)
    if use_ard:
        lengthscale_constraint = Interval(0.01, 0.5)
    else:
        lengthscale_constraint = Interval(
            0.01, sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
    # outputscale_constraint = Interval(0.05, 20.0)
    outputscale_constraint = Interval(0.5, 5.)

    # add in temporal dimension if t is not None
    if train_t is not None and time_varying:
        train_x = torch.hstack((train_t.reshape(-1, 1), train_x))

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(
        device=train_x.device, dtype=train_y.dtype)

    kern = CasmoKernel(cs=configspace, lamda=0.5, ard=use_ard,
                       time_varying=time_varying,
                       continuous_lengthscale_constraint=lengthscale_constraint,
                       categorical_lengthscale_constraint=lengthscale_constraint,
                       lengthscale_scaling=lengthscale_scaling)

    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        kern=kern,
        outputscale_constraint=outputscale_constraint,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    loaded_hypers = False
    # if hyperparameters are already supplied, no need to optimize GP
    if hypers is not None and len(hypers):
        try:
            model.load_state_dict(hypers)
            loaded_hypers = True
        except Exception as e:
            logging.warning(
                f'Exception={e} occurred when loading the hyperparameters of the GP. Now training from scratch!')

    if not loaded_hypers:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = np.sqrt(0.01 * 0.5)
        hypers["likelihood.noise"] = noise_variance if noise_variance is not None else 0.005
        model.initialize(**hypers)

        # Use the adam optimizer
        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.2)

        for _ in range(num_steps):
            optimizer.zero_grad()
            output = model(train_x, )
            try:
                loss = -mll(output, train_y).float()
                loss.backward()
                optimizer.step()
                if verbose and _ % 50 == 0:
                    logging.info(
                        f'Optimising GP log-likelihood: Iter={_}, Loss={loss.detach().numpy()}')

            except Exception as e:
                print(
                    f'RuntimeError={e} occurred due to non psd covariance matrix. returning the model at last successful iter')
                model.eval()
                likelihood.eval()
                return model
    # Switch to eval mode
    model.eval()
    likelihood.eval()
    if return_hypers:
        return model, model.state_dict()
    else:
        return model


def normalize(data, wrt):
    """ Normalize data to be in range (0,1), with respect to (wrt) boundaries,
        which can be specified.
    """
    return (data - np.min(wrt, axis=0)) / (
        np.max(wrt, axis=0) - np.min(wrt, axis=0))


def get_dim_info(cs: CS.ConfigurationSpace, x, return_indices=False):
    """Return the information on the categorical, integer and continuous spaces"""
    if len(cs) != len(x):
        # this is because x is longer than cs -- the final dimensions are the contextual info presented as fixed dimensions.
        x = deepcopy(x)[:len(cs)]
    cat_dims, cont_dims, int_dims = [], [], []
    cat_dims_idx, cont_dims_idx, int_dims_idx = [], [], []
    for i, variable in enumerate(range(len(x))):
        # do not sample an inactivated hyperparameter (such a hyperparameter has nan value imputed)
        if x[variable] != x[variable]:
            continue
        if type(cs[cs.get_hyperparameter_by_idx(variable)]) == CSH.CategoricalHyperparameter:
            cat_dims.append(cs.get_hyperparameter_by_idx(variable))
            cat_dims_idx.append(i)
        elif type(cs[cs.get_hyperparameter_by_idx(variable)]) in [CSH.UniformIntegerHyperparameter,
                                                                  CSH.NormalIntegerHyperparameter]:
            int_dims.append(cs.get_hyperparameter_by_idx(variable))
            int_dims_idx.append(i)
        elif type(cs[cs.get_hyperparameter_by_idx(variable)]) in [CSH.UniformFloatHyperparameter,
                                                                  CSH.NormalFloatHyperparameter]:
            cont_dims.append(cs.get_hyperparameter_by_idx(variable))
            cont_dims_idx.append(i)
    if return_indices:
        return cat_dims_idx, cont_dims_idx, int_dims_idx
    return cat_dims, cont_dims, int_dims


def sample_discrete_neighbour(cs: CS.ConfigurationSpace, x, frozen_dims: List[int] = None):
    """Sample a neighbour from x in one of the active hyperparameter.

    select type:

    frozen_dims: the frozen dimensions where neighbours that differ from them will be rejected.

    """
    # note that for acquisition function optimisation (which this def is likely used), integer-type variables are treated
    # as discrete.
    assert len(x) >= len(cs)
    if len(x) > len(cs):
        # this is because x is longer than cs -- the final dimensions are the contextual info presented as fixed dimensions.
        fixed_dims = x[len(cs):]
        x = deepcopy(x)[:len(cs)]
    else:
        fixed_dims = None
    cat_dims, _, int_dims = get_dim_info(cs, x)
    config = CS.Configuration(cs, vector=x.detach(
    ).numpy() if isinstance(x, torch.Tensor) else x)

    try:
        cs.check_configuration(config)
    except ValueError as e:
        # there seems to be a bug with ConfigSpace that raises error even when a config is valid
        # Issue #196: https://github.com/automl/ConfigSpace/issues/196
        # print(config)
        config = CS.Configuration(cs, config.get_dictionary())

    # print(config)
    config_pert = deepcopy(config)
    selected_dim = str(np.random.choice(cat_dims + int_dims, 1)[0])
    index_in_array = cs.get_idx_by_hyperparameter_name(selected_dim)
    while config_pert[selected_dim] is None or (frozen_dims is not None and index_in_array in frozen_dims):
        selected_dim = str(np.random.choice(cat_dims + int_dims, 1)[0])
        index_in_array = cs.get_idx_by_hyperparameter_name(selected_dim)

    # if the selected dimension is categorical, change the value to another variable
    if selected_dim in cat_dims:
        config_pert[selected_dim] = np.random.choice(
            cs[selected_dim].choices, 1)[0]
        while config_pert[selected_dim] == config[selected_dim]:
            config_pert[selected_dim] = np.random.choice(
                cs[selected_dim].choices, 1)[0]
    elif selected_dim in int_dims:
        lb, ub = cs[selected_dim].lower, cs[selected_dim].upper
        if selected_dim in ['NAS_policy_num_layers', 'NAS_q_num_layers']:
            candidates = list(
                {max(lb, config[selected_dim] - 1), min(ub, config[selected_dim] + 1)})
        else:
            candidates = list(range(max(lb, min(config[selected_dim] - 1, round(config[selected_dim] * 0.8))),
                                    min(ub, max(round(config[selected_dim] * 1.2), config[selected_dim] + 1)) + 1))
        config_pert[selected_dim] = np.random.choice(candidates, 1)[0]
        while config_pert[selected_dim] == config[selected_dim]:
            config_pert[selected_dim] = np.random.choice(candidates, 1)[0]
    config_pert = deactivate_inactive_hyperparameters(config_pert, cs)
    x_pert = config_pert.get_array()
    if fixed_dims is not None:
        x_pert = np.concatenate([x_pert, fixed_dims])
    return x_pert


def hamming_distance(x1, x2, normalize=False):
    diff = (x1 != x2).to(dtype=torch.float)
    dist = diff.sum()
    if normalize:
        dist /= x1.shape[0]
    return dist


def construct_bounding_box(cs: CS.ConfigurationSpace, x, tr_length, weights=None, ):
    """Construct a bounding box around x_cont with tr_length being the k-dimensional trust region size.
    The weights should be the learnt lengthscales of the GP surrogate model.
    """
    if weights is None:
        weights = 1. / len(x.shape[0]) * np.ones(x.shape[1])
    # non-ard lengthscales passed -- this must be a scalar input
    elif len(weights) != x.shape[0]:
        weights = weights * np.ones(x.shape[0])
    weights = weights / weights.mean()
    # We now have weights.prod() = 1
    weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
    lb, ub = np.zeros_like(x), np.ones_like(x)
    for i, dim in enumerate(x):
        if np.isnan(x[i]) or i >= len(cs):
            lb[i], ub[i] = 0., 1.
        else:
            hp = cs[cs.get_hyperparameter_by_idx(i)]
            if type(hp) == CSH.CategoricalHyperparameter:
                lb[i], ub[i] = 0, len(hp.choices)
            else:
                lb[i] = np.clip(x[i] - weights[i] * tr_length / 2.0, 0.0, 1.0)
                ub[i] = np.clip(x[i] + weights[i] * tr_length / 2.0, 0.0, 1.0)
                if type(hp) in [CSH.UniformIntegerHyperparameter, CSH.NormalIntegerHyperparameter,
                                CSH.NormalFloatHyperparameter, CSH.UniformFloatHyperparameter]:
                    lb[i] = max(hp._inverse_transform(hp.lower), lb[i])
                    ub[i] = min(hp._inverse_transform(hp.upper), ub[i])
    return lb, ub


def get_start_point(cs: CS.ConfigurationSpace, x_center, frozen_dims: List[int] = None, return_config=False):
    # get a perturbed starting point from x_center
    new_config_array = deepcopy(x_center)

    perturbation_factor = [0.8, 1.2]  # <- taken from PB2
    for i in range(new_config_array.shape[0]):
        # print(param_name)
        if np.isnan(new_config_array[i]) or (frozen_dims is not None and i in frozen_dims):
            continue
        param_name = cs.get_hyperparameter_by_idx(i)
        if type(cs[param_name]) == CSH.CategoricalHyperparameter:
            new_config_array[i] = np.random.choice(
                range(len(cs[param_name].choices)))
        elif type(cs[param_name]) in [CSH.UniformIntegerHyperparameter, CSH.NormalIntegerHyperparameter] and cs[param_name].lower == 0 and cs[param_name].upper == 1:   # binary
            new_config_array[i] = np.random.choice([0, 1])
        else:
            perturbation = np.random.uniform(*perturbation_factor)
            new_config_array[i] = np.clip(x_center[i] * perturbation, 0., 1.)
    # whether to change the network depth
    config = CS.Configuration(cs, vector=new_config_array)
    config = deactivate_inactive_hyperparameters(config, cs)

    try:
        cs.check_configuration(config)
    except ValueError:
        config = CS.Configuration(cs, config.get_dictionary())
    new_config_array = config.get_array()
    if return_config:
        return new_config_array, config
    return new_config_array


def grad_search(cs: CS.ConfigurationSpace,
                x_center,
                f: Callable,
                n_restart: int = 1,
                step: int = 40,
                batch_size: int = 1,
                dtype=torch.float,
                fixed_dims=None,
                verbose: bool = True,
                ):
    """Vanilla gradient-based search"""
    num_fixed_dims = x_center.shape[0] - \
        len(cs) if x_center.shape[0] > len(cs) else 0
    if num_fixed_dims > 0:
        fixed_dims = list(range(len(cs), x_center.shape[0]))
    else:
        fixed_dims = None

    x0s = []
    for _ in range(n_restart):
        p = cs.sample_configuration().get_array()
        if fixed_dims is not None:
            p = np.concatenate((p, x_center[fixed_dims]))
        x0s.append(p)
    x0 = np.array(x0s).astype(np.float)

    def _grad_search(x0):
        lb, ub = np.zeros(x0.shape[0]), np.ones(x0.shape[0])
        n_step = 0
        x = deepcopy(x0)
        acq_x = f(x).detach().numpy()
        x_tensor = torch.tensor(x, dtype=dtype).requires_grad_(True)
        optimizer = torch.optim.Adam([{"params": x_tensor}], lr=0.1)

        while n_step <= step:
            optimizer.zero_grad()
            acq = f(x_tensor)
            acq.backward()
            if num_fixed_dims:
                x_tensor.grad[fixed_dims] = 0.
            if verbose and n_step % 20 == 0:
                logging.info(
                    f'Acquisition optimisation: Step={n_step}: Value={x_tensor}. Acq={acq_x}.')
            optimizer.step()
            with torch.no_grad():
                x_tensor = torch.maximum(torch.minimum(x_tensor, torch.tensor(
                    ub).to(x_tensor.dtype)), torch.tensor(lb).to(x_tensor.dtype))
            n_step += 1
        x = x_tensor.detach().numpy().astype(x0.dtype)
        acq_x = f(x).detach().numpy()
        del x_tensor
        return x, acq_x

    X, fX = [], []
    for i in range(n_restart):
        res = _grad_search(x0[i, :])
        X.append(res[0])
        fX.append(res[1])
    top_idices = np.argpartition(
        np.array(fX).flatten(), batch_size)[:batch_size]
    return np.array([x for i, x in enumerate(X) if i in top_idices]).astype(np.float), np.array(fX).astype(np.float).flatten()[top_idices]


def interleaved_search(cs: CS.ConfigurationSpace,
                       n_dim,
                       x_center,
                       f: Callable,
                       max_dist_cont: float,
                       max_dist_cat: float = None,
                       cont_int_lengthscales: float = None,
                       n_restart: int = 1,
                       step: int = 40,
                       batch_size: int = 1,
                       interval: int = 1,
                       dtype=torch.float,
                       frozen_dims: List[int] = None,
                       frozen_vals: list = None,
                       num_fixed_dims: int = None,
                       verbose: bool = True
                       ):
    """
    x_center: the previous best x location that will be the centre of the bounding box
    f: the objective function of the interleaved_search. In this case, it is usually the acquisition function.
        This objective should be minimized.
    max_dist_cont: the bounding box length of the continuous trust region
    max_dist_cat: the bounding box length of the categorical trust region. This is in terms of normalized Hamming distance >0, <=1.
    cont_int_lengthscales: the lengthscales of the learnt GP model on the continuous and integer dimensions
    n_restart: number of restarts for the acquisition function optimization.
    """
    # when a x_center with a higher dimension than that specified by he configspace object, the additional dimensions
    #   are treated as "contextual" dimensions which are fixed during acquisition optimization.
    if max_dist_cat is None:
        max_dist_cat = 1.  # the normalized hamming distance is upper bounded by 1.
    num_fixed_dims = n_dim - len(cs) if n_dim > len(cs) else 0
    if num_fixed_dims > 0:
        fixed_dims = list(range(len(cs), n_dim))
    else:
        fixed_dims = None

    cat_dims, cont_dims, int_dims = get_dim_info(
        cs, cs.sample_configuration().get_array(), return_indices=True)

    if x_center is not None:
        assert x_center.shape[0] == n_dim
        x_center_fixed = deepcopy(
            x_center[-num_fixed_dims:]) if num_fixed_dims > 0 else None

        # generate the initially random points by perturbing slightly from the best location
        x_center_local = deepcopy(x_center)
        if frozen_dims is not None:
            x_center_local[frozen_dims] = frozen_vals   # freeze these values
        x0s = []
        lb, ub = construct_bounding_box(
            cs, x_center_local, max_dist_cont, cont_int_lengthscales)
        for _ in range(n_restart):
            if num_fixed_dims:
                p = get_start_point(
                    cs, x_center_local[:-num_fixed_dims], frozen_dims=frozen_dims)
                p = np.concatenate((p, x_center_fixed))
            else:
                p = get_start_point(cs, x_center_local,
                                    frozen_dims=frozen_dims)
            x0s.append(p)
    else:
        lb, ub = np.zeros(n_dim), np.ones(n_dim)
        x0s = [cs.sample_configuration().get_array() for _ in range(n_restart)]
        x_center_fixed = None

    x0 = np.array(x0s).astype(np.float)  # otherwise error on jade

    def _interleaved_search(x0):
        x = deepcopy(x0)
        acq_x = f(x).detach().numpy()
        n_step = 0
        while n_step <= step:
            # First optimise the continuous part, freezing the categorical part
            x_tensor = torch.tensor(x, dtype=dtype).requires_grad_(True)

            optimizer = torch.optim.Adam([{"params": x_tensor}], lr=0.1)
            for _ in range(interval):
                optimizer.zero_grad()
                acq = f(x_tensor)
                acq.backward()
                # freeze the grads of the non-continuous dimensions & the fixed dims
                for n, w in enumerate(x_tensor):
                    if n not in cont_dims or (fixed_dims is not None and n in fixed_dims) or (
                            frozen_dims is not None and n in frozen_dims):
                        x_tensor.grad[n] = 0.
                if verbose and n_step % 20 == 0:
                    logging.info(
                        f'Acquisition optimisation: Step={n_step}: Value={x_tensor}. Acq={acq_x}.')
                optimizer.step()
                with torch.no_grad():
                    x_nan_mask = torch.isnan(x_tensor)
                    # replace the data from the optimized tensor
                    x_tensor[cont_dims] = torch.maximum(torch.minimum(x_tensor[cont_dims], torch.tensor(ub[cont_dims])),
                                                        torch.tensor(lb[cont_dims])).to(x_tensor.dtype)
                    # enforces the nan entries remain nan
                    x_tensor[x_nan_mask] = torch.tensor(
                        np.nan, dtype=x_tensor.dtype)

                    # fixed dimensions should not be updated during the optimization here. Enforce the constraint below
                    if x_center_fixed is not None:
                        # the fixed dimensions are not updated according to the gradient information.
                        x_tensor[-num_fixed_dims:] = torch.tensor(
                            x_center_fixed, dtype=dtype)
                    # print(x_tensor)

            x = x_tensor.detach().numpy().astype(x0.dtype)
            del x_tensor

            # Then freeze the continuous part and optimise the categorical part
            for j in range(interval):
                neighbours = [sample_discrete_neighbour(
                    cs, x, frozen_dims=frozen_dims) for _ in range(10)]
                for i, neighbour in enumerate(neighbours):
                    neighbours[i][int_dims] = np.clip(
                        neighbour[int_dims], lb[int_dims], ub[int_dims])
                acq_x = f(x).detach().numpy()
                acq_neighbour = np.array(
                    [f(n).detach().numpy() for n in neighbours]).astype(np.float)
                acq_neighbour_argmin = np.argmin(acq_neighbour)
                acq_neighbour_min = acq_neighbour[acq_neighbour_argmin]
                if acq_neighbour_min < acq_x:
                    x = deepcopy(neighbours[acq_neighbour_argmin])
                    acq_x = acq_neighbour_min
            n_step += interval
        return x, acq_x

    def local_search(x):
        acq = np.inf
        x = deepcopy(x)
        logging.info(f'Bounds: {lb}, {ub}')

        if x_center_fixed is not None:
            x_config = CS.Configuration(cs, vector=x[:-num_fixed_dims])
        else:
            x_config = CS.Configuration(cs, vector=x)
        for _ in range(step):
            n_config = CS.util.get_random_neighbor(
                x_config, seed=int(np.random.randint(10000)))
            n = n_config.get_array()
            if x_center_fixed is not None:
                # the fixed dimensions are not updated according to the gradient information.
                n = np.concatenate((n, x_center_fixed))
            n = np.clip(n, lb, ub)
            acq_ = f(n).detach().numpy()
            if acq_ < acq:
                acq = acq_
                x = n
                x_config = n_config
        return x, acq

    X, fX = [], []
    for i in range(n_restart):
        res = _interleaved_search(x0[i, :])
        X.append(res[0])
        fX.append(res[1])
    top_idices = np.argpartition(
        np.array(fX).flatten(), batch_size)[:batch_size]
    return np.array([x for i, x in enumerate(X) if i in top_idices]).astype(np.float), np.array(fX).astype(np.float).flatten()[top_idices]
