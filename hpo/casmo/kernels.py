import gpytorch.kernels
import torch
from gpytorch.kernels import Kernel
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from gpytorch.kernels.kernel import default_postprocess_script
import math
from gpytorch.constraints import Interval


class CategoricalOverlap(Kernel):
    """Implementation of the categorical overlap kernel.
    This is the most basic form of the categorical kernel that essentially invokes a Kronecker delta function
    between any two elements.
    """
    has_lengthscale = True

    def __init__(self, **kwargs):
        super(CategoricalOverlap, self).__init__(
            has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        with torch.no_grad():  # discrete kernels are not differentiable. Make is explicit as such
            # First, convert one-hot to ordinal representation
            diff = x1[:, None] - x2[None, :]
            # nonzero location = different cat
            diff[torch.abs(diff) > 1e-5] = 1
            # invert, to now count same cats
            diff1 = torch.logical_not(diff).float()
            if self.ard_num_dims is not None and self.ard_num_dims > 1:
                k_cat = torch.sum(self.lengthscale * diff1,
                                  dim=-1) / torch.sum(self.lengthscale)
            else:
                # dividing by number of cat variables to keep this term in range [0,1]
                k_cat = torch.sum(diff1, dim=-1) / x1.shape[1]
            if diag:
                return torch.diag(k_cat).float()
            return k_cat.float()


class ExpCategoricalOverlap(CategoricalOverlap):
    """
    Exponentiated categorical overlap kernel
    $$ k(x, x') = \\exp(\frac{\\lambda}{n}) \\sum_{i=1}^n [x_i = x'_i] )$$ (if non-ARD)
    or
    $$ k(x, x') = \\exp(\frac{1}{n} \\sum_{i=1}^n \\lambda_i [x_i = x'_i]) $$ if ARD
    """
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp='rbf', **params):
        with torch.no_grad():  # discrete kernels are not differentiable. Make is explicit as such
            diff = x1[:, None] - x2[None, :]
            diff[torch.abs(diff) > 1e-5] = 1
            diff1 = torch.logical_not(diff).float()

            def rbf(d, ard):
                if ard:
                    return torch.exp(torch.sum(d * self.lengthscale, dim=-1) / torch.sum(self.lengthscale))
                else:
                    return torch.exp(self.lengthscale * torch.sum(d, dim=-1) / x1.shape[1])

            if exp == 'rbf':
                k_cat = rbf(
                    diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
            else:
                raise ValueError(
                    'Exponentiation scheme %s is not recognised!' % exp)
            if diag:
                return torch.diag(k_cat).float()
        return k_cat.float()


class L1Distance(torch.nn.Module):
    """Compute L1 distance between two input vectors"""

    def __init__(self, postprocess_script=default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    def _dist(self, x1, x2, postprocess, x1_eq_x2=False):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

        # Compute l1 distance
        res = (x1.unsqueeze(1) - x2.unsqueeze(0)).abs().sum(-1)
        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
            res.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Zero out negative values
        res.clamp_min_(0)
        return self._postprocess(res) if postprocess else res


class TemporalKernel(Kernel):
    """Kernel function to compute L1 distance between two vectors, without a lengthscale.
    This is useful for computing the distance between the time vectors in time-varying GP
    surrogate.

    epsilon (epsilon) is the "forgetting" parameter of the time-varying GP.
    """
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TemporalKernel, self).__init__(**kwargs)
        self.distance_module = L1Distance()
        eps_constraint = Interval(0., 1.)
        self.register_parameter(
            name='raw_epsilon', parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint('raw_epsilon', eps_constraint)

    @property
    def epsilon(self):
        return self.raw_epsilon_constraint.transform(self.raw_epsilon)

    @epsilon.setter
    def epsilon(self, value):
        self._set_epsilon(value)

    def _set_epsilon(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_eps)
        self.initialize(
            raw_eps=self.raw_eps_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):

        dist = self.covar_dist(
            x1, x2, diag=diag, **params
        )
        time_ker = (1. - self.epsilon) ** (0.5 * dist)
        time_ker_diag = torch.diag(time_ker)
        if diag:
            return time_ker_diag
        return time_ker


class ConditionalMatern(gpytorch.kernels.MaternKernel):
    has_lengthscale = True

    def __init__(self, cs: CS.ConfigurationSpace, nu=2.5, **kwargs):
        self.cs = cs
        super().__init__(nu, **kwargs)

    def forward(self, x1, x2,
                diag=False,
                **params):
        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
        x1_ = (x1 - mean).div(self.lengthscale)
        x2_ = (x2 - mean).div(self.lengthscale)
        distance = self.covar_dist(x1_, x2_, diag=diag, **params)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)
        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (
                math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
        else:
            raise RuntimeError(
                f'nu must be in {0.5, 1.5, 2.5} but got {self.nu}!')
        return constant_component * exp_component


class CasmoKernel(Kernel):
    """Implementation of the kernel in Casmopolitan"""
    has_lengthscale = True

    def __init__(self, cs: CS.ConfigurationSpace, lamda=0.5, ard=True,
                 lengthscale_scaling=3.,
                 time_varying=False,
                 categorical_lengthscale_constraint=None,
                 continuous_lengthscale_constraint=None,
                 **kwargs):
        """
        Note that the integer dimensions are treated as continuous here (but as discrete during acquisition optimization).
        No explicit wrapping of the integer dimensions are required, as the samples are generated from local search
        (which always produces a valid configuration on the integer vertices).
        """
        super().__init__(has_lengthscale=True, **kwargs)
        self.cs = cs
        self.dim = len(self.cs.get_hyperparameters())
        self.lengthscale_scaling = lengthscale_scaling
        self.continuous_lengthscale_constraint = continuous_lengthscale_constraint
        self.lamda_ = lamda
        self.ard = ard
        # extract the dim indices of the continuous dimensions (incl. integers)
        self.cont_dims = [i for i, dim in enumerate(self.cs.get_hyperparameters())
                          if type(dim) in [CSH.UniformIntegerHyperparameter, CSH.UniformFloatHyperparameter]]
        self.cat_dims = [i for i, dim in enumerate(self.cs.get_hyperparameters()) if
                         type(dim) == CSH.CategoricalHyperparameter]

        # initialise the kernels
        self.continuous_kern = ConditionalMatern(cs=self.cs, nu=2.5, ard_num_dims=len(self.cont_dims) if ard else None,
                                                 lengthscale_scaling=lengthscale_scaling,
                                                 lengthscale_constraint=continuous_lengthscale_constraint)
        self.categorical_kern = ExpCategoricalOverlap(ard_num_dims=len(self.cat_dims) if ard else None,
                                                      lengthscale_constraint=categorical_lengthscale_constraint)
        self.time_varying = time_varying
        self.time_kernel = TemporalKernel() if time_varying else None

    def _set_lamda(self, value):
        self.lamda_ = max(0., min(value, 1.))

    @property
    def lamda(self):
        return self.lamda_

    @lamda.setter
    def lamda(self, value):
        self._set_lamda(value)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x1_mask=None, x2_mask=None,
                diag=False, last_dim_is_batch=False, **params):
        """
        todo: for now the masking is only available for the integer/continuous dimensions. This works for now as
            none of the categorical variables is currently conditional. If and when we have conditional categoricals,
            the categorical kernels need to be amended correspondingly to avoid problems.
        """
        assert x1.shape[1] >= self.dim, f'Dimension mismatch! Expected = {self.dim} but got {x1.shape[1]}'
        # it is possible for x1.shape[1] to be larger than self.dim, due to auxiliary dimensions that are not part of
        #   the active configspace but provide extra information about the search space. These are appended to the end
        #   of the vector, and the cont_dims are changed accordingly (assuming these additional dimensions are all
        #   continuous)

        # WARNING: any additional contextual information MUST be added to the END of the vector. If it is present
        #   anywhere else, the kernel may give incorrect results, WITHOUT raising an exception.
        if self.time_varying:
            x1, t1 = x1[:, 1:], x1[:, :1]
            if x2 is not None:
                x2, t2 = x2[:, 1:], x2[:, :1]
            else:
                t2 = None
        else:
            t1 = t2 = None
        if x1.shape[1] > self.dim:
            self.continuous_kern = ConditionalMatern(cs=self.cs, nu=2.5,
                                                     ard_num_dims=x1.shape[1] if self.ard else None,
                                                     lengthscale_scaling=self.lengthscale_scaling,
                                                     lengthscale_constraint=self.continuous_lengthscale_constraint)
            self.cont_dims += list(range(self.dim, x1.shape[1]))
            self.dim = x1.shape[1]

        if x2 is not None:
            assert x2.shape[1] == x1.shape[1]
        if t1 is not None and self.time_kernel is not None:
            assert t1.shape[0] == x1.shape[
                0], f'Dimension mismatch between x1 {x1.shape[0]} and its timestep vector t1 {t1.shape[0]}!'
        if t2 is not None and self.time_kernel is not None:
            assert t2.shape[0] == x2.shape[0], f'Dimension mismatch between x2 and its timestep vector t2!'
        if len(self.cat_dims) == 0 and len(self.cont_dims) == 0:
            raise ValueError("Zero-dimensioned problem!")
        elif len(self.cat_dims) > 0 and len(self.cont_dims) == 0:  # entirely categorical
            spatial_ker_val = self.categorical_kern.forward(
                x1, x2, diag=diag, **params)
        elif len(self.cont_dims) > 0 and len(self.cat_dims) == 0:  # entirely continuous
            spatial_ker_val = self.continuous_kern.forward(x1, x2, diag=diag, x1_mask=x1_mask, x2_mask=x2_mask,
                                                           **params)
        else:
            # mixed case
            x1_cont, x2_cont = x1[:, self.cont_dims], x2[:, self.cont_dims]
            x1_cat, x2_cat = x1[:, self.cat_dims], x2[:, self.cat_dims]
            spatial_ker_val = (1. - self.lamda) * (self.categorical_kern.forward(x1_cat, x2_cat, diag=diag, **params) +
                                                   self.continuous_kern.forward(x1_cont, x2_cont, x1_mask=x1_mask,
                                                                                x2_mask=x2_mask,
                                                                                diag=diag, **params)) + \
                self.lamda * self.categorical_kern.forward(x1_cat, x2_cat, diag=diag, **params) * \
                self.continuous_kern.forward(x1_cont, x2_cont, x1_mask=x1_mask, x2_mask=x2_mask,
                                             diag=diag,
                                             **params)

        if self.time_kernel is None or t1 is None or t2 is None:
            ker_val = spatial_ker_val
        else:  # product kernel between the temporal and spatial kernel values.
            ker_val = self.time_kernel.forward(t1, t2) * spatial_ker_val
        return ker_val
