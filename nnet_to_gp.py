
import torch, os
import numpy as np
import pickle as pkl
from tqdm import trange
from keras.datasets.mnist import load_data as mnist

class VAE(torch.nn.Module):
    def __init__(self, n, d, q, m):
        super().__init__()
        self.q = q
        self.encoder_ha = torch.nn.Linear(d, m)
        self.encoder_hb = torch.nn.Linear(m, m)
        self.encoder_ho = torch.nn.Linear(m, q*2)

        self.decoder_ha = torch.nn.Linear(q, m)
        self.decoder_ho = torch.nn.Linear(m, d)

    def encode(self, Y):
        e = self.encoder_ha(Y).relu()
        e = self.encoder_hb(e).relu()
        e = self.encoder_ho(e).tanh()
        q = len(e.T)//2

        qX = torch.distributions.Normal(
            loc=e[:, :q],
            scale=e[:, q:]*0.5 + 0.5 + 1e-6)
        
        return qX

    def decode(self, X):
        d = self.decoder_ha(X).relu()
        return self.decoder_ho(d)

def nnet_to_gp(vae, d, q, m):
    p = torch.nn.Parameter

    W_recon = torch.cat([vae.decoder_ha.weight, vae.decoder_ha.bias[..., None]], axis=1)
    gp = GP(d=len(Y.T), q=q+1, m=m, W_init=W_recon)
    gp.n_approx = 8

    def refresh(gp): gp(torch.ones(2, q+1))

    gp.intercept.constant = p(vae.decoder_ho.bias.reshape(-1, 1))
    refresh(gp)  # calculate C_uu

    jit = torch.eye(len(gp.C_uu))*2e-3
    w_norm = gp.W.norm(dim=1).diag()
    gp.variational_strategy._variational_distribution.variational_mean = p(\
        vae.decoder_ho.weight @ w_norm @ (gp.C_uu + jit).cholesky())
    refresh(gp)

    return gp

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.ion(); plt.style.use('ggplot')
    torch.manual_seed(42)
    np.random.seed(42)

    (Y, l), (_, _) = mnist()
    Y = torch.tensor(Y/Y.max()).round().float()
    Y = Y.reshape(len(Y), np.prod(Y.shape[1:]))[np.isin(l, [0, 1]), :]

    n, d = Y.shape
    q, m = 2, 30

    vae = VAE(n, d, q, m)

    if torch.cuda.is_available():
        vae = vae.cuda()
        Y = Y.cuda()

    if os.path.exists('vae_params.pkl'):
        with open('vae_params.pkl', 'rb') as file:
            vae.load_state_dict(pkl.load(file))
    else:
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
        pX = torch.distributions.Normal(0., 1.)

        batch_size = n; steps = 100000; losses = []; iterator = trange(steps)
        for i in iterator:
            optimizer.zero_grad()

            idx = np.random.choice(range(n), batch_size, replace=False); idx.sort()
            qX = vae.encode(Y[idx, :])
            X = qX.rsample()
            probs = vae.decode(X).sigmoid()
            likelihood = torch.distributions.Bernoulli(probs=probs).log_prob(Y[idx, :]).sum()

            neg_elbo = torch.distributions.kl_divergence(qX, pX).sum()/batch_size - likelihood/batch_size

            losses.append(neg_elbo.item())
            iterator.set_description(
                'ELBO:' + str(np.round(-neg_elbo.item(), 2)) +'; ' + \
                'Step:' + str(i)
            )
            neg_elbo.backward()
            optimizer.step()

        with open('vae_params.pkl', 'wb') as file:
            pkl.dump(vae.cpu().state_dict(), file)

        samps = torch.distributions.Bernoulli(probs=probs.cpu()).sample()
        plt.imshow(samps[np.random.choice(range(batch_size)), :].reshape(28, 28))

    #################################
    # gplvm approx

    # Set activation to relu in activation.py to match the nnet.

    from utils import GP, pred_nn
    X = vae.encode(Y).loc.detach()
    X = torch.cat([X, torch.ones(len(X), 1)], axis=1)

    gp = nnet_to_gp(vae, d, q, m)
    probs = gp(X[:10, :]).loc.T.detach().sigmoid()
    samps = torch.distributions.Bernoulli(probs=probs).sample()

    plt.imshow(samps[np.random.choice(range(len(samps))), :].reshape(28, 28))

    # check if the redefining the nnet from the gp produces good images
    probs_nn = pred_nn(X, d, m, gp).sigmoid()
    samps = torch.distributions.Bernoulli(probs=probs_nn).sample()

    plt.imshow(samps[np.random.choice(range(len(samps))), :].reshape(28, 28))
