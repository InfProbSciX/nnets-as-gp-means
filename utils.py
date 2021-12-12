
import torch
import numpy as np
from tqdm import trange
from functools import lru_cache
from scipy.integrate import quad
from scipy.special import gegenbauer, gamma

from gpytorch.means import ConstantMean
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import LinearKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP

from activation import activation, torchify, Softplus

@lru_cache(maxsize=100)
def gegenbauer_coefs(*args): return gegenbauer(*args)

def C(n, a, x):
    C = gegenbauer_coefs(n, a)
    if type(x) is not torch.Tensor: x = torch.tensor(x)
    x = [x[None, ...]**(C.order - order) * coef \
         for order, coef in enumerate(C.coef)]
    return torch.sum(torch.cat(x, axis=0), axis=0)

class Constants:
    @staticmethod
    def alpha(d):
        return (d - 2)/2

    @staticmethod
    def omega(d):
        return gamma(d/2) / (gamma(d/2 - 0.5) * np.sqrt(np.pi))

@torchify
def kernel(x):
    x[x.abs() > 1] = x[x.abs() > 1].sign()
    t = x.arccos()
    return (t.sin() + t.cos() * (np.pi - t)) / np.pi

class Eigenvalues:
    @staticmethod
    def _integrand(x, shape, n, d):
        return shape(x) * C(n, Constants.alpha(d), x) * (1 - x**2)**((d - 3)/2)

    @staticmethod
    @lru_cache(maxsize=100)
    def _estm_integral(shape, n, d):
        integral = quad(Eigenvalues._integrand, -1, 1, args=(shape, n, d))[0]
        return Constants.omega(d) * integral / C(n, Constants.alpha(d), 1.)

    @staticmethod
    def activation(n, d):
        return Eigenvalues._estm_integral(activation, n, d)

    @staticmethod
    def kernel(n, d):
        return Eigenvalues._estm_integral(kernel, n, d)

def inducing_covariance(r, d, n_approx_for_sum):
    C_uu = 0.0
    for n in range(n_approx_for_sum):
        lmbda = Eigenvalues.kernel(n, d)
        if abs(lmbda) > 1e-9:
            alpha = Constants.alpha(d)
            sigma = Eigenvalues.activation(n, d)
            C_uu += ((sigma/lmbda)*sigma * (n + alpha) * C(n, alpha, r)) / (alpha)
    return C_uu

def cross_covariance(r, d, n_approx_for_sum):
    C_uf = 0.0
    for n in range(n_approx_for_sum):
        alpha = Constants.alpha(d)
        sigma = Eigenvalues.activation(n, d)
        C_uf += (sigma * (n + alpha) * C(n, alpha, r))/alpha
    return C_uf

class GP(ApproximateGP):
    def __init__(self, d, q, m, n_approx_for_sum=8, W_init=None, cross_covariance='relu'):
        self.W = torch.randn(m, q) if W_init is None else W_init
        q_u = CholeskyVariationalDistribution(m, batch_shape=(d,))
        q_f = VariationalStrategy(self, self.W, q_u, learn_inducing_locations=False)

        super(GP, self).__init__(q_f)

        self.intercept = ConstantMean(batch_shape=(d,))
        self.n_approx_for_sum = n_approx_for_sum
        self.cross_covariance = cross_covariance

    def forward(self, X):
        intercept = self.intercept(X)
        W = X[:len(self.W), :]
        X = X[len(self.W):, :]

        X_norm = X.norm(dim=1)[..., None]; X_normed = X / X_norm
        W_norm = W.norm(dim=1)[..., None]; W_normed = W / W_norm

        d = len(X.T)

        if self.cross_covariance == 'relu':
            C_uf = X_norm.T * cross_covariance(W_normed @ X_normed.T, d, self.n_approx_for_sum)
        elif self.cross_covariance == 'softplus':
            C_uf = X_norm.T * activation(W_normed @ X_normed.T)
        else:
            raise ValueError('Unrecognized activation')

        C_ff = (X_norm * X_norm.T) * kernel(X_normed @ X_normed.T) + torch.eye(len(X)) * 1e0
        C_uu = inducing_covariance(W_normed @ W_normed.T, d, self.n_approx_for_sum)

        S = torch.cat([
                torch.cat([C_uu  , C_uf], axis=1),
                torch.cat([C_uf.T, C_ff], axis=1)
            ], axis=0)
        S = S + torch.eye(len(S)) * 1e-3

        self.C_uu = C_uu
        return MultivariateNormal(intercept, S)

def pred_nn(X, d, m, gp):
    W = gp.W

    X_norm = X.norm(dim=1)[..., None]
    W_norm = W.norm(dim=1)[..., None]

    X_normed = X / X_norm
    W_normed = W / W_norm

    mu = gp.intercept(torch.zeros(1, 1)).reshape(-1)
    u = gp.variational_strategy.variational_distribution.mean.T

    j_m = torch.eye(len(gp.C_uu))*1e-3
    p = torch.nn.Parameter
    w_scale = (1/W_norm).reshape(-1).diag()

    hidden_layer = torch.nn.Linear(len(X.T), m, bias=False)
    obs_layer = torch.nn.Linear(m, d, bias=True)

    hidden_layer.weight = p(W)
    obs_layer.bias = p(mu)
    obs_layer.weight = p(u.T @ (gp.C_uu + j_m).cholesky().inverse() @ w_scale)
    return obs_layer(activation(hidden_layer(X))).detach()
