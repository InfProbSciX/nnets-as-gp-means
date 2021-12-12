
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood

from utils import GP, torchify, Softplus, pred_nn

plt.ion(); plt.style.use('ggplot')
torch.manual_seed(42); np.random.seed(42)

# Set activation to softplus in activation.py. This defines
# a non-singular cross covariance as opposed to the relu.

if __name__ == '__main__':

    X_orig = torch.tensor([[-2.06, -2.30, -1.10, -1.07, -1.10,  1.74,  1.45,  1.14,  1.14,  1.62]]).T
    Y = torch.tensor([[-2.15, -1.89, -1.03, -1.35, -1.41, -1.94, -1.52, -1.28, -1.15, -1.34]]).T

    # embed the 1d data in 3d
    t = np.pi / 6
    R = torch.tensor([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).float()

    X = torch.cat([X_orig, torch.ones_like(X_orig)], axis=1) @ R
    X = torch.cat([X, torch.ones(len(X), 1)], axis=1)

    # define the GP and fit it
    n = len(Y); m = 36; d = 3
    gp = GP(d=len(Y.T), q=d, m=m, cross_covariance='softplus')
    likelihood = GaussianLikelihood()

    elbo_func = VariationalELBO(likelihood, gp, num_data=n)
    optimizer = torch.optim.Adam([
        dict(params=gp.parameters(), lr=0.02),
        dict(params=likelihood.parameters(), lr=0.02)
    ])

    steps = 500; losses = []; iterator = trange(steps)
    for i in iterator:
        optimizer.zero_grad()
        neg_elbo = -elbo_func(gp(X), Y.T).sum()

        losses.append(neg_elbo.item())
        iterator.set_description(
            'ELBO:' + str(np.round(-neg_elbo.item(), 2)) +'; ' + \
            'Step:' + str(i)
        )
        neg_elbo.backward()
        optimizer.step()

    X_predict_orig = torch.linspace(-2.5, 2, 50)
    X_predict = torch.cat([X_predict_orig[..., None], torch.ones(len(X_predict_orig), 1)], axis=1) @ R
    X_predict = torch.cat([X_predict, torch.ones(len(X_predict), 1)], axis=1)
    dist = gp(X_predict)

    Y_predict = dist.loc[0].detach()

    plt.plot(X_predict_orig, Y_predict, label='GP prediction')
    plt.scatter(X_orig[:, 0], Y[:, 0], label='data')
    plt.plot(X_predict_orig, pred_nn(X_predict, len(Y.T), m, gp)[:, 0], label='NNet prediction')
    plt.legend()

    # using the relu covariance here results in better uncertainties
    # where the covariance isn't singular
